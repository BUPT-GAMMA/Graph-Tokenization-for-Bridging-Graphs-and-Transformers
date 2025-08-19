"""
清理后的图序列化器基类 - 专注于token管理和序列化接口
======================================================

这是一个清理后的基类，专注于token管理和序列化接口，移除了所有混乱的残留函数。
"""

from collections import defaultdict
from math import log10
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
import dgl
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.base_loader import BaseDataLoader
from utils.logger import get_logger
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import threading

logger = get_logger(__name__)


# ============== 多进程批量序列化（fork-only） ==============
# 说明：为满足“内部并发、上层无感知”的诉求，在 Linux/posix 下使用 fork 共享内存态，
# 无需 pickle 序列化器实例与 DGLGraph 对象，避免高额拷贝成本。
# 非 fork 平台不提供隐式回退（遵循不做回退的规则）。
_MP_GLOBAL_SERIALIZER = None  # type: Optional["BaseGraphSerializer"]
_MP_GRAPH_DATA_LIST = None    # type: Optional[List[Dict[str, Any]]]
_MP_STATS_GRAPH_LIST = None   # type: Optional[List[Dict[str, Any]]]


def _mp_init_worker(serializer_ref, graph_data_ref):
    global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
    _MP_GLOBAL_SERIALIZER = serializer_ref
    _MP_GRAPH_DATA_LIST = graph_data_ref


def _mp_process_range(lo_hi_kwargs: Tuple[int, int, Dict[str, Any]]):
    lo, hi, kwargs = lo_hi_kwargs
    ser = _MP_GLOBAL_SERIALIZER
    lst = _MP_GRAPH_DATA_LIST
    assert ser is not None and lst is not None, "多进程序列化未正确初始化"
    out: List[Tuple[int, SerializationResult]] = []
    for i in range(lo, hi):
        res = ser.serialize(lst[i], **kwargs)
        out.append((i, res))
    return out

# ============== batch_multiple_serialize 专用多进程 worker ==============

def _mp_multiple_process_range(args: Tuple[int, int, int, Dict[str, Any]]):
    """多进程区间处理：调用 multiple_serialize。"""
    lo, hi, num_samples, kwargs = args
    ser = _MP_GLOBAL_SERIALIZER
    lst = _MP_GRAPH_DATA_LIST
    assert ser is not None and lst is not None, "多进程序列化未正确初始化"
    out: List[Tuple[int, SerializationResult]] = []
    for i in range(lo, hi):
        res = ser.multiple_serialize(lst[i], num_samples=num_samples, parallel=False, **kwargs)
        out.append((i, res))
    return out

# ============== 多进程统计收集（fork-only） ==============
_MP_STATS_SERIALIZER = None  # type: Optional["BaseGraphSerializer"]
_MP_STATS_GRAPH_LIST = None  # type: Optional[List[Dict[str, Any]]]


def _mp_stats_init(serializer_ref, graph_data_ref):
    global _MP_STATS_SERIALIZER, _MP_STATS_GRAPH_LIST
    _MP_STATS_SERIALIZER = serializer_ref
    _MP_STATS_GRAPH_LIST = graph_data_ref


def _mp_stats_max_range(lo_hi: Tuple[int, int]) -> Tuple[int, int]:
    lo, hi = lo_hi
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    max_node = _torch.tensor(0, dtype=_torch.long)
    max_edge = _torch.tensor(0, dtype=_torch.long)
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"图数据缺少必需字段 'dgl_graph': 索引 {i}"
        g = lst[i]['dgl_graph']
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if node_type_ids.numel() > 0:
            max_node = _torch.maximum(max_node, node_type_ids.max())
        if edge_type_ids.numel() > 0:
            max_edge = _torch.maximum(max_edge, edge_type_ids.max())
    # 返回“最大 id + 1”作为维度提示
    return int(max_node.item()) + 1 if max_node.item() >= 0 else 0, int(max_edge.item()) + 1 if max_edge.item() >= 0 else 0


def _mp_stats_count_range(args: Tuple[int, int, int, int]) -> "torch.Tensor":
    lo, hi, Vdim0, Edim = args
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    Vdim1 = Vdim0
    flat_len = Vdim0 * Edim * Vdim1
    # 使用 int32 降低内存占用；bincount 输出为 int64，后续转为 int32 再累加
    flat = _torch.zeros(flat_len, dtype=_torch.int32)
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"图数据缺少必需字段 'dgl_graph': 索引 {i}"
        g = lst[i]['dgl_graph']
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if edge_type_ids.numel() == 0:
            continue
        src, dst = ser._get_all_edges_from_heterograph(g)
        src_t = node_type_ids.index_select(0, src).long()
        dst_t = node_type_ids.index_select(0, dst).long()
        et_t = edge_type_ids.long()
        # 边界检查
        if src_t.numel() == 0:
            continue
        if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
            raise ValueError("统计计数阶段出现类型ID越界，请检查维度推断逻辑。")
        lin = src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t
        cnt = _torch.bincount(lin, minlength=flat_len)
        flat.add_(cnt.to(_torch.int32))
    return flat


def _mp_stats_count_range_sparse(args: Tuple[int, int, int, int]) -> Dict[int, int]:
    """多进程统计（稀疏）：返回线性键 -> 计数 的字典。
    为避免构建致密张量，此路径仅对出现过的键累加。
    """
    lo, hi, Vdim0, Edim = args
    ser = _MP_STATS_SERIALIZER
    lst = _MP_STATS_GRAPH_LIST
    assert ser is not None and lst is not None
    import torch as _torch
    Vdim1 = Vdim0
    counts: Dict[int, int] = {}
    for i in range(lo, hi):
        assert 'dgl_graph' in lst[i], f"图数据缺少必需字段 'dgl_graph': 索引 {i}"
        g = lst[i]['dgl_graph']
        edge_type_ids = ser._dataset_loader.get_graph_edge_type_ids(g)
        if edge_type_ids.numel() == 0:
            continue
        node_type_ids = ser._dataset_loader.get_graph_node_type_ids(g)
        src, dst = ser._get_all_edges_from_heterograph(g)
        src_t = node_type_ids.index_select(0, src).long()
        dst_t = node_type_ids.index_select(0, dst).long()
        et_t = edge_type_ids.long()
        if src_t.numel() == 0:
            continue
        if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
            raise ValueError("统计计数阶段出现类型ID越界，请检查维度推断逻辑。")
        lin = (src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t).to(_torch.long)
        if lin.numel() == 0:
            continue
        lin_sorted, _ = _torch.sort(lin)
        uniq, cts = _torch.unique_consecutive(lin_sorted, return_counts=True)
        for k, v in zip(uniq.tolist(), cts.tolist()):
            counts[k] = counts.get(k, 0) + int(v)
    return counts

class GlobalIDMapping:
    """全局ID映射管理器"""
    def __init__(self, dgl_graph: dgl.DGLGraph):
        self.global_to_local = {}  # 全局ID -> (节点类型, 局部ID)
        self.local_to_global = {}  # (节点类型, 局部ID) -> 全局ID
        self.current_global_id = 0
        
        # 构建映射
        for ntype in dgl_graph.ntypes:
            num_nodes = dgl_graph.num_nodes(ntype)
            for local_id in range(num_nodes):
                self.global_to_local[self.current_global_id] = (ntype, local_id)
                self.local_to_global[(ntype, local_id)] = self.current_global_id
                self.current_global_id += 1
    
    def to_global_id(self, ntype: str, local_id: int) -> int:
        """将局部ID转换为全局ID"""
        return self.local_to_global.get((ntype, local_id))
    
    def to_local_id(self, global_id: int) -> Tuple[str, int]:
        """将全局ID转换为(节点类型, 局部ID)"""
        return self.global_to_local.get(global_id)
    
    def get_total_nodes(self) -> int:
        """获取总节点数"""
        return self.current_global_id


class SerializationResult:
    """序列化结果 - 统一格式"""
    def __init__(self, token_sequences: List[List[int]], element_sequences: List[List[str]], id_mapping: GlobalIDMapping):
        assert token_sequences and element_sequences, "token_sequences和element_sequences不能为空"
        
        assert len(token_sequences) == len(element_sequences), "token_sequences和element_sequences长度必须相同"
        
        # 验证每个序列的长度匹配
        for i, (token_seq, element_seq) in enumerate(zip(token_sequences, element_sequences)):
            assert len(token_seq) == len(element_seq), f"第{i}个序列中token和element长度不匹配: {len(token_seq)} vs {len(element_seq)}"
        
        self.token_sequences = token_sequences  # 多个token序列
        self.element_sequences = element_sequences  # 对应的图元素序列
        self.id_mapping = id_mapping
    
    def get_sequence_count(self) -> int:
        """获取序列数量"""
        return len(self.token_sequences)
    
    def get_sequence(self, idx: int) -> Tuple[List[int], List[str]]:
        """获取指定索引的序列
        
        Args:
            idx: 序列索引
            
        Returns:
            Tuple[List[int], List[str]]: (token_ids, element_ids)
            
        Raises:
            IndexError: 索引超出范围
        """
        assert 0 <= idx < len(self.token_sequences), f"序列索引{idx}超出范围[0, {len(self.token_sequences)}]"
        
        return (self.token_sequences[idx], self.element_sequences[idx])
    
    def get_element_info(self, idx: int) -> Tuple[str, str, int]:
        """获取第一个序列中某个位置对应元素的完整信息
        
        Args:
            idx: 元素在第一个序列中的位置
        
        Returns:
            Tuple[元素类型, 元素ID, token]
            
        Raises:
            IndexError: 索引超出范围
        """
        assert 0 <= idx < len(self.element_sequences[0]), f"元素索引{idx}超出第一个序列范围[0, {len(self.element_sequences[0])}]"
        
        element_id = self.element_sequences[0][idx]
        token = self.token_sequences[0][idx]
        
        # 解析元素类型
        if element_id.startswith('node_'):
            element_type = 'node'
        elif element_id.startswith('edge_'):
            element_type = 'edge'
        else:
            element_type = 'unknown'
        
        return (element_type, element_id, token)
    
    def get_graph_feature(self, idx: int, dgl_graph, feature_name: str):
        """根据element_id获取DGL图中对应元素的特征
        
        Args:
            idx: 元素在第一个序列中的位置
            dgl_graph: DGL图对象
            feature_name: 特征名称（如'feat', 'atomic_num', 'bond_type'等）
            
        Returns:
            对应的特征值，可能是标量或向量
            
        Raises:
            IndexError: 索引超出范围
            ValueError: 无法解析element_id或获取特征
        """
        element_type, element_id, token = self.get_element_info(idx)
        
        try:
            if element_type == 'node':
                # 解析节点ID
                node_id = int(element_id.split('_')[1])
                if feature_name in dgl_graph.ndata:
                    feature_tensor = dgl_graph.ndata[feature_name][node_id]
                    # 如果是标量，返回Python数值；如果是向量，返回numpy数组
                    if feature_tensor.dim() == 0:
                        return feature_tensor.item()
                    else:
                        return feature_tensor.detach().cpu().numpy().tolist()
                else:
                    raise ValueError(f"图中不存在节点特征 '{feature_name}'")
                    
            elif element_type == 'edge':
                # 解析边ID
                edge_id = int(element_id.split('_')[1])
                if feature_name in dgl_graph.edata:
                    feature_tensor = dgl_graph.edata[feature_name][edge_id]
                    # 如果是标量，返回Python数值；如果是向量，返回numpy数组
                    if feature_tensor.dim() == 0:
                        return feature_tensor.item()
                    else:
                        return feature_tensor.detach().cpu().numpy().tolist()
                else:
                    raise ValueError(f"图中不存在边特征 '{feature_name}'")
            else:
                raise ValueError(f"未知的元素类型: {element_type}")
                
        except (IndexError, ValueError) as e:
            raise ValueError(f"无法获取元素 {element_id} 的特征 {feature_name}: {str(e)}")


class BaseGraphSerializer(ABC):
    """图序列化器基类 - 统一接口规范"""
    
    def __init__(self):
        """初始化基础序列化器"""
        self.id_mapper = None
        self.name = "base_serializer"
        self._initialized = False
        self._dataset_loader: BaseDataLoader = None
        self._dataset_stats = {}
        # 边ID映射表缓存
        self._current_edge_id_mapping = None
        # 线程本地存储，保证并发时互不干扰
        self._thread_local = threading.local()
        # 数值型三元组频率张量及其维度 (V, E, V)
        self._triplet_frequency_tensor = None
        self._triplet_tensor_dims = None
        # 稀疏三元组统计（线性键 -> 计数），以及稀疏键、值向量缓存
        self._triplet_sparse_counts: Dict[int, int] = {}
        self._triplet_sparse_keys = None  # torch.Tensor[int64], sorted
        self._triplet_sparse_vals = None  # torch.Tensor[int32]
        # 并发与统计配置（成员级控制，不通过方法传参）
        self.stats_parallel_enabled: bool = True
        self.parallel_num: int = max(os.cpu_count()//2, 1)
        self.enable_string_stats: bool = False  # 默认关闭字符串统计热路径
    
    # ==================== 统一接口规范 ====================
    
    def initialize_with_dataset(self, dataset_loader: BaseDataLoader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        使用数据集初始化序列化器（统一接口）
        
        Args:
            dataset_loader: 数据集加载器，提供节点token映射等功能
            graph_data_list: 图数据列表，用于收集统计信息（可选）
        """
        self._dataset_loader = dataset_loader
        self._dataset_stats = {}
        # self._dataset_loader.load_data()
        
        # 确保数据加载器的缓存已构建
        # if hasattr(dataset_loader, 'ensure_cache_built'):
        #     dataset_loader.ensure_cache_built(graph_data_list)
        
        # 获取并保存数据集级别的token映射
        # if hasattr(dataset_loader, 'get_node_token_map'):
        #     self._dataset_node_token_map = dataset_loader.get_node_token_map()
        # else:
        #     raise ValueError("❌ 数据加载器必须实现 get_node_token_map 方法")
            
        # if hasattr(dataset_loader, 'get_edge_token_map'):
        #     self._dataset_edge_token_map = dataset_loader.get_edge_token_map()
        # else:
        #     raise ValueError("❌ 数据加载器必须实现 get_edge_token_map 方法")
        
        # 调用子类的具体初始化逻辑
        self._initialize_serializer(dataset_loader, graph_data_list)
        
        self.most_frequent_edge_type = self._dataset_loader.get_most_frequent_edge_type()
        
        self._initialized = True
        logger.info(f"✅ {self.name} 序列化器初始化完成")
    
    def serialize(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        序列化单个图（统一接口）
        
        Args:
            graph_data: 图数据，包含dgl_graph等字段
            **kwargs: 额外的序列化参数
            
        Returns:
            SerializationResult: 序列化结果
        """
        assert self._initialized, f"❌ {self.name} 序列化器尚未初始化，请先调用 initialize_with_dataset"
        
        assert 'dgl_graph' in graph_data and graph_data['dgl_graph'] is not None, "❌ 图数据缺少有效的dgl_graph字段"
        
        dgl_graph = graph_data['dgl_graph']
        assert dgl_graph.num_nodes() > 0, "❌ 图没有节点，无法进行序列化"
        
        # 为当前线程设置边ID映射表，避免并发冲突
        setattr(self._thread_local, 'current_edge_id_mapping', self._build_edge_id_mapping(dgl_graph))
        
        # 调用子类的具体序列化逻辑
        return self._serialize_single_graph(graph_data, **kwargs)
    
    def multiple_serialize(self, graph_data: Dict[str, Any], num_samples: int = 1, *, parallel: bool = False,**kwargs) -> SerializationResult:
        """
        对单个图进行多次序列化（统一接口）
        
        Args:
            graph_data: 图数据，包含dgl_graph等字段
            num_samples: 采样次数，选择不同起始点进行多次序列化
            **kwargs: 额外的序列化参数
            
        Returns:
            SerializationResult: 包含多个序列的结果
        """
        if 'dgl_graph' not in graph_data or graph_data['dgl_graph'] is None:
            raise ValueError("❌ 图数据缺少有效的dgl_graph字段")
        
        dgl_graph = graph_data['dgl_graph']
        
        if dgl_graph.num_nodes() == 0:
            raise ValueError("❌ 图没有节点，无法进行序列化")
        
        # 创建全局ID映射
        id_mapping = GlobalIDMapping(dgl_graph)
        total_nodes = id_mapping.get_total_nodes()
        
        
        # 确保采样次数不超过节点数
        actual_samples = min(num_samples, total_nodes)
        
        token_sequences: List[List[int]] = []
        element_sequences: List[List[str]] = []
        
        # 选择不同的起始节点进行多次序列化
        if actual_samples == 1:
            # 单次采样：从节点0开始
            start_nodes = [0]
        else:
            # 多次采样：均匀分布选择起始节点
            if total_nodes == 1:
                start_nodes = [0] * actual_samples
            else:
                step = max(1, total_nodes // actual_samples)
                start_nodes = [(i * step) % total_nodes for i in range(actual_samples)]
        
        # 执行多次序列化
        if not parallel or actual_samples <= 1:
            for start_node in start_nodes:
                result = self.serialize(graph_data, start_node=start_node, **kwargs)
                if result and result.get_sequence_count() > 0:
                    first_token_seq, first_element_seq = result.get_sequence(0)
                    if first_token_seq:
                        token_sequences.append(first_token_seq)
                        element_sequences.append(first_element_seq)
        else:
            # 并行：同一实例 + 线程本地状态，避免实例级共享数据冲突
            # 并行度采用内置自动检测的 CPU 核心数，忽略外部配置
            futures = []  # type: list
            with ThreadPoolExecutor(max_workers=self.parallel_num) as executor:
                for start_node in start_nodes:
                    def _make_task(sn: int):
                        def _task() -> Tuple[List[int], List[str]]:
                            res = self.serialize(graph_data, start_node=sn, **kwargs)
                            ts, es = res.get_sequence(0)
                            return ts, es
                        return _task
                    futures.append(executor.submit(_make_task(start_node)))

                for fut in futures:
                    ts, es = fut.result()
                    token_sequences.append(ts)
                    element_sequences.append(es)
        
        if not token_sequences:
            return SerializationResult([[]], [[]], id_mapping)
        
        return SerializationResult(token_sequences, element_sequences, id_mapping)
    
    def batch_serialize(self, graph_data_list: List[Dict[str, Any]], desc: str = None, *, parallel: bool = True, **kwargs) -> List[SerializationResult]:
        """
        批量序列化多个图（统一接口）
        
        说明：parallel=True 时，仅使用“多进程（fork）”并发；不再提供线程并发路径。
        
        Args:
            graph_data_list: 图数据列表
            desc: 进度条描述
            **kwargs: 额外的序列化参数
            
        Returns:
            List[SerializationResult]: 批量序列化结果列表
        """
        if desc is None:
            desc = f"🔄 {self.name}批量序列化"
        
        if not parallel:
            results: List[SerializationResult] = []
            success_count = 0
            with tqdm(total=len(graph_data_list), desc=desc,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i, graph_data in enumerate(graph_data_list):
                    result = self.serialize(graph_data, **kwargs)
                    results.append(result)
                    success_count += 1
                    pbar.set_postfix({'成功': success_count, '失败': 0})
                    pbar.update(1)
            return results
        else:
            # 并行：使用多进程（fork），严格保序；不提供线程并发路径
            # 并行度采用内置自动检测的 CPU 核心数，忽略外部配置
            n = len(graph_data_list)
            if n == 0:
                return []
            workers = min(self.parallel_num, n)
            # 切分为 workers 个分片
            chunk_sizes = [(n // workers) + (1 if i < (n % workers) else 0) for i in range(workers)]
            indices = []
            start = 0
            for sz in chunk_sizes:
                indices.append((start, start + sz))
                start += sz

            results: List[Optional[SerializationResult]] = [None] * n

            # 始终使用 fork 上下文创建本地进程池（即使外层为 spawn）
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                raise RuntimeError("多进程序列化要求 fork（仅限 Linux/posix）；当前环境不支持 fork")
            global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
            _MP_GLOBAL_SERIALIZER = self
            _MP_GRAPH_DATA_LIST = graph_data_list
            with ctx.Pool(processes=workers, initializer=_mp_init_worker, initargs=(self, graph_data_list)) as pool:
                for pairs in pool.imap_unordered(_mp_process_range, [(lo, hi, kwargs) for (lo, hi) in indices if lo < hi], chunksize=1):
                    for i, res in pairs:
                        results[i] = res
            return [r for r in results if r is not None]
    
    def batch_multiple_serialize(self, graph_data_list: List[Dict[str, Any]], num_samples: int = 1, desc: str = None, *, parallel: bool = True, **kwargs) -> List[SerializationResult]: 
        """
        批量对多个图进行多次序列化（统一接口）
        
        Args:
            graph_data_list: 图数据列表
            num_samples: 每个图的采样次数
            desc: 进度条描述
            **kwargs: 额外的序列化参数
            
        Returns:
            List[SerializationResult]: 批量多次序列化结果列表
        """
        if desc is None:
            desc = f"🔄 {self.name}批量多次序列化"
        
        if not parallel:
            results: List[SerializationResult] = []
            success_count = 0
            with tqdm(total=len(graph_data_list), desc=desc,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for i, graph_data in enumerate(graph_data_list):
                    result = self.multiple_serialize(graph_data, num_samples=num_samples, **kwargs)
                    results.append(result)
                    success_count += 1
                    pbar.set_postfix({'成功': success_count, '失败': 0})
                    pbar.update(1)
            return results
        else:
            # 并行：使用多进程（fork），严格保序；不提供线程并发路径
            n = len(graph_data_list)
            if n == 0:
                return []
            workers = min(self.parallel_num, n)

            # 切分为 workers 个分片
            chunk_sizes = [(n // workers) + (1 if i < (n % workers) else 0) for i in range(workers)]
            indices = []
            start = 0
            for sz in chunk_sizes:
                indices.append((start, start + sz))
                start += sz

            results: List[Optional[SerializationResult]] = [None] * n

            # 始终使用 fork 上下文创建本地进程池（即使外层为 spawn）
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                raise RuntimeError("多进程序列化要求 fork（仅限 Linux/posix）；当前环境不支持 fork")

            global _MP_GLOBAL_SERIALIZER, _MP_GRAPH_DATA_LIST
            _MP_GLOBAL_SERIALIZER = self
            _MP_GRAPH_DATA_LIST = graph_data_list

            with ctx.Pool(processes=workers, initializer=_mp_init_worker, initargs=(self, graph_data_list)) as pool:
                it_args = [(lo, hi, num_samples, kwargs) for (lo, hi) in indices if lo < hi]
                for pairs in pool.imap_unordered(_mp_multiple_process_range, it_args, chunksize=1):
                    for i, res in pairs:
                        results[i] = res

            return [r for r in results if r is not None]
    
    # ==================== 子类必须实现的抽象方法 ====================
    
    @abstractmethod
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        子类实现的具体初始化逻辑
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表，用于统计信息收集
        """
        pass
    
    @abstractmethod
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        子类实现的单个图序列化逻辑
        
        Args:
            graph_data: 图数据
            **kwargs: 额外的序列化参数
            
        Returns:
            SerializationResult: 序列化结果
        """
        pass
    
    # ==================== 工具方法 ====================
    
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """
        获取单个节点的token ID
        
        Args:
            sample: 图数据样本
            node_id: 节点ID
            
        Returns:
            int: 节点的token ID
        """
        assert self._initialized, "❌ 序列化器尚未初始化"
          
        return self._dataset_loader.get_node_token(graph, node_id, ntype)
    
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        """
        获取单个边的token ID
        
        Args:
            sample: 图数据样本
            edge_id: 边ID
            
        Returns:
            int: 边的token ID
        """
        assert self._initialized, "❌ 序列化器尚未初始化"
        
        return self._dataset_loader.get_edge_token(graph, edge_id, etype)
    
    
    def tokens_to_string(self, token_ids: List[int]) -> str:
        """
        将token ID序列转换为字符串表示（统一接口）
        
        Args:
            token_ids: token ID序列
            
        Returns:
            字符串表示
            
        Raises:
            ValueError: 当序列化器未初始化或缺少token映射时抛出
        """
        assert self._initialized, "序列化器未初始化"
        assert hasattr(self._dataset_loader, 'get_token_readable'), "数据集加载器缺少 get_token_readable 方法"
        
        return ''.join([self._dataset_loader.get_token_readable(token_id) for token_id in token_ids])
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return self._dataset_stats.copy()
    
    def is_initialized(self) -> bool:
        """检查序列化器是否已初始化"""
        return self._initialized
    
    # ==================== 兼容性方法 ====================
    
    def deserialize(self, result: SerializationResult, **kwargs) -> dgl.DGLGraph:
        """从序列化结果重建图结构（兼容性方法）
        
        Args:
            result: 序列化结果
            **kwargs: 额外的反序列化参数
            
        Returns:
            dgl.DGLGraph: 重建的图
            
        Raises:
            NotImplementedError: 大多数序列化器不支持反序列化
        """
        raise NotImplementedError("此序列化器不支持反序列化功能")
    
    # ====================== 序列化公共方法 ======================
    
    
    def _validate_graph_data(self, mol_data: Dict[str, Any],) -> dgl.DGLGraph:
        """
        验证图数据并返回DGL图（公共方法）
        
        Args:
            mol_data: 图数据
            method_name: 方法名称（用于错误信息）
            
        Returns:
            dgl.DGLGraph: 验证后的DGL图
            
        Raises:
            ValueError: 如果数据无效
        """
        assert 'dgl_graph' in mol_data, "❌ 输入数据缺少'dgl_graph'字段"
        
        assert mol_data['dgl_graph'].num_nodes() > 0, "❌ 图没有节点，无法进行序列化"
        
        return mol_data['dgl_graph']
    
    
    def _get_all_edges_from_heterograph(self, dgl_graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从异构图中获取所有边，兼容同构图和异构图（公共方法）
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (src_nodes, dst_nodes)
        """
        if len(dgl_graph.etypes) == 1:
            # 同构图或单一边类型，直接获取
            return dgl_graph.edges()
        else:
            # 异构图，需要遍历所有边类型
            all_src = []
            all_dst = []
            
            for etype in dgl_graph.etypes:
                src, dst = dgl_graph.edges(etype=etype)
                all_src.append(src)
                all_dst.append(dst)
            
            # 合并所有边
            if all_src:
                import torch
                combined_src = torch.cat(all_src, dim=0)
                combined_dst = torch.cat(all_dst, dim=0) 
                return combined_src, combined_dst
            else:
                raise ValueError("❌ 图没有边")
    
    def _build_edge_id_mapping(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], int]:
        """
        构建边ID映射表，用于优化边ID查询
        
        Args:
            dgl_graph: DGL图
            
        Returns:
            Dict[Tuple[int, int], int]: (src, dst) -> edge_id 的映射表
        """
        edge_mapping = {}
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        dgl_graph.edge_ids
        # 对于DGL图，边ID通常就是边的索引位置
        # 我们直接使用索引位置作为边ID，避免重复调用edge_ids
        for i, (src, dst) in enumerate(zip(src_nodes.numpy(), dst_nodes.numpy())):
            src, dst = int(src), int(dst)
            edge_mapping[(src, dst)] = i
        
        return edge_mapping
    
    def _get_edge_id(self, dgl_graph: dgl.DGLGraph, src: int, dst: int) -> int:
        """
        获取DGL图中边的ID（公共方法）
        
        Args:
            dgl_graph: DGL图
            src: 源节点
            dst: 目标节点
            
        Returns:
            int: 边ID，如果找不到返回-1
        """
        # 优先使用线程本地缓存的边ID映射表
        thread_mapping = getattr(self._thread_local, 'current_edge_id_mapping', None)
        if thread_mapping is not None:
            edge_id = thread_mapping.get((src, dst))
            assert edge_id is not None, f"❌ 找不到边ID: ({src}, {dst})"
            return edge_id
        # 兼容旧路径：使用实例级映射（不建议在并行下使用）
        if self._current_edge_id_mapping is not None:
            edge_id = self._current_edge_id_mapping.get((src, dst))
            assert edge_id is not None, f"❌ 找不到边ID: ({src}, {dst})"
            return edge_id
        
        # 回退到原始方法
        try:
            edge_ids = dgl_graph.edge_ids(src, dst)
            return int(edge_ids)
        except Exception as e:
            raise ValueError(f"❌ 获取边ID失败 ({src}->{dst}): {e}")
      
    def _get_edge_type(self, dgl_graph: dgl.DGLGraph, src: int, dst: int) -> str:
        """
        获取DGL图中边的类型（公共方法）
        
        Args:
            dgl_graph: DGL图
            src: 源节点 
            dst: 目标节点
            
        Returns:
            str: 边类型
        """
        edge_id = self._get_edge_id(dgl_graph, src, dst)
        edge_type = self._dataset_loader.get_edge_type(dgl_graph, edge_id=edge_id)
        return edge_type
      
    def _get_node_type(self, dgl_graph: dgl.DGLGraph, node_id: int) -> str:
        """
        获取DGL图中节点的类型（公共方法）
        """
        node_type = self._dataset_loader.get_node_type(dgl_graph, node_id)
        return node_type
    
    
    def _build_adjacency_list_from_dgl(self, dgl_graph: dgl.DGLGraph) -> List[List[int]]:
        """
        从DGL图构建邻接列表（公共方法）
        
        对于欧拉回路序列化，每条DGL边都代表一次可遍历的路径。
        如果DGL图包含双向边 (u,v) 和 (v,u)，则在邻接列表中：
        - adj_list[u] 包含 v（表示可以从u走到v）
        - adj_list[v] 包含 u（表示可以从v走到u）
        这样每个节点的度数就是它在所有邻接列表中出现的总次数。
        
        Args:
            dgl_graph: DGL图
            
        Returns:
            List[List[int]]: 邻接列表，每条DGL边对应一个邻接关系
        """
        num_nodes = dgl_graph.num_nodes()
        adj_list = [[] for _ in range(num_nodes)]
        
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # 直接按照DGL图中的每条边构建邻接列表
        # 不需要检查是否双向，每条边都是一次遍历机会
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            adj_list[src].append(dst)
        
        return adj_list
      
    def _convert_path_to_tokens(self, node_path: List[int], mol_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """将路径转换为 token 与元素序列（张量化取数，末端转 list）。"""
        if not node_path:
            return [], []

        import torch as _torch

        dgl_graph = mol_data['dgl_graph']

        # -------- 节点 tokens（整图张量 + index_select） --------
        node_ids_tensor: torch.Tensor = _torch.as_tensor([int(n) for n in node_path], dtype=_torch.long)
        node_tok_2d: torch.Tensor = self._dataset_loader.get_graph_node_token_ids(dgl_graph)  # [N, Dn]
        selected_node_tok: torch.Tensor = node_tok_2d.index_select(0, node_ids_tensor)  # [P, Dn]

        # -------- 边 ids（整图 + 一次性计算 + 可选掩码） --------
        P = int(node_ids_tensor.shape[0])
        include_edges = bool(getattr(self, 'include_edge_tokens', True) and P > 1)
        if include_edges:
            src_tensor = node_ids_tensor[:-1]
            dst_tensor = node_ids_tensor[1:]
            try:
                edge_ids_tensor_full = dgl_graph.edge_ids(src_tensor, dst_tensor)  # [P-1]
            except Exception as e:
                # 兼容多重边（同一 (u,v) 存在多条边）或异构情况导致的标量转换失败
                # 回退到逐对查询，取首个匹配的边ID，以确保稳定且可序列化
                import torch as _torch
                eid_list: List[int] = []
                src_list = src_tensor.tolist()
                dst_list = dst_tensor.tolist()
                for u, v in zip(src_list, dst_list):
                    try:
                        # 单对查询：若存在多条边，DGL 返回 1D 张量；我们取第一个
                        eids = dgl_graph.edge_ids(int(u), int(v), return_uv=False)
                        if hasattr(eids, 'numel') and eids.numel() > 1:
                            eid_list.append(int(eids[0].item()))
                        else:
                            eid_list.append(int(eids.item()))
                    except Exception:
                        # 最后兜底：构建 (u,v)->eid 映射，选第一个遍历到的边
                        mapping = self._build_edge_id_mapping(dgl_graph)
                        eid = mapping.get((int(u), int(v)))
                        if eid is None:
                            raise ValueError(f"获取边ID失败: (src={u}, dst={v}) 无对应边")
                        eid_list.append(int(eid))
                edge_ids_tensor_full = _torch.as_tensor(eid_list, dtype=_torch.long)

            # 掩码：省略最高频边
            keep_mask = None
            if getattr(self, 'omit_most_frequent_edge', False):
                edge_type_ids_all = self._dataset_loader.get_graph_edge_type_ids(dgl_graph)  # [E]
                edge_type_ids_seq = edge_type_ids_all.index_select(0, edge_ids_tensor_full)  # [P-1]
                mf_name = getattr(self, 'most_frequent_edge_type', None)
                if mf_name is not None:
                    mf_id = int(self._dataset_loader.get_edge_type_id_by_name(mf_name))
                    keep_mask = edge_type_ids_seq.ne(mf_id)

            if keep_mask is None:
                edge_ids_tensor = edge_ids_tensor_full
                keep_mask_bool = _torch.ones(edge_ids_tensor_full.shape[0], dtype=_torch.bool)
            else:
                edge_ids_tensor = edge_ids_tensor_full[keep_mask]
                keep_mask_bool = keep_mask
            Ke = int(edge_ids_tensor.shape[0])
        else:
            edge_ids_tensor = _torch.empty((0,), dtype=_torch.long)
            keep_mask_bool = _torch.empty((0,), dtype=_torch.bool)
            Ke = 0

        # -------- 节点/边 tokens 张量化获取 --------
        Dn = int(selected_node_tok.shape[1])
        if include_edges:
            edge_tok_2d_all = self._dataset_loader.get_graph_edge_token_ids(dgl_graph)  # [E, De]
            # 直接复用已计算的 edge_ids_tensor_full，避免重复查询
            selected_edge_tok_all = edge_tok_2d_all.index_select(0, edge_ids_tensor_full)  # [P-1, De]
            if keep_mask_bool.numel() > 0 and keep_mask_bool.sum().item() != keep_mask_bool.numel():
                selected_edge_tok = selected_edge_tok_all[keep_mask_bool]
            else:
                selected_edge_tok = selected_edge_tok_all
            De = int(selected_edge_tok.shape[1]) if selected_edge_tok.numel() > 0 else int(edge_tok_2d_all.shape[1])
        else:
            # 没有边
            De = 1
            selected_edge_tok = _torch.empty((0, De), dtype=_torch.long)

        # -------- 输出顺序：交错（node_i token 后接 edge_i,i+1 token），这是原标准顺序 --------
        # 快速路径：Dn==1 且（无边或 De==1）
        if Dn == 1 and (not include_edges or De == 1):
            P = int(selected_node_tok.shape[0])
            L = P + Ke
            seq = _torch.empty(L, dtype=_torch.long)
            # 节点位置：i + 已保留边的前缀和
            if include_edges and keep_mask_bool.numel() > 0:
                shift = _torch.cat([_torch.zeros(1, dtype=_torch.long), keep_mask_bool.cumsum(0)])
            else:
                shift = _torch.zeros(P, dtype=_torch.long)
            pos_nodes = _torch.arange(P, dtype=_torch.long) + shift
            node_flat = selected_node_tok.view(-1).to(_torch.long)  # 确保数据类型匹配
            seq[pos_nodes] = node_flat
            if include_edges and Ke > 0:
                pos_edges = pos_nodes[:-1][keep_mask_bool] + 1
                seq[pos_edges] = selected_edge_tok.view(-1).to(_torch.long)  # 确保数据类型匹配
            token_list = seq.tolist()
            element_list = [""] * len(token_list)
            return token_list, element_list

        # 通用路径：任意 Dn/De
        out_chunks = []
        node_rows = selected_node_tok.tolist()
        edge_rows = selected_edge_tok.tolist() if (include_edges and Ke>0) else []
        e_ptr = 0
        for i in range(len(node_rows)):
            out_chunks.extend(node_rows[i])
            if i < len(node_rows)-1 and (include_edges and Ke>0):
                if bool(keep_mask_bool[i].item()):
                    out_chunks.extend(edge_rows[e_ptr])
                    e_ptr += 1
        token_list = out_chunks
        element_list = [""] * len(token_list)
        return token_list, element_list
      
    def _convert_path_to_tokens_old(self, node_path: List[int], mol_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """将边路径转换为token序列和元素序列
        
        Returns:
            Tuple[List[int], List[str]]: (token_sequence, element_sequence)
        """
        
        if not node_path:
            return [], []
        dgl_graph = mol_data['dgl_graph']
        
        result_token_ids = []
        result_element_ids = []  # 元素序列：直接对应原图元素ID
        
        # 构建包含节点和边的序列
        for i, node in enumerate(node_path):
            # 添加节点token和元素信息
            node_tokens = self.get_node_token(dgl_graph, node)
            result_token_ids.extend(node_tokens)
            
            # 为每个token添加对应的element信息
            if len(node_tokens) > 1:
                result_element_ids.append(f"START_NODE_{node}")
                for j, token in enumerate(node_tokens[1:-1]):
                      result_element_ids.append(f"node_{node}_dim_{j}")
                result_element_ids.append(f"END_NODE_{node}")
            else:
                result_element_ids.append(f"node_{node}")
                
            if not self.include_edge_tokens:
                continue
            
            # 如果不是最后一个节点，添加边token和元素信息
            if i >= len(node_path) - 1:
              continue
            
            next_node = node_path[i + 1]
            edge_id = self._get_edge_id(dgl_graph, node, next_node)
            edge_tokens = self.get_edge_token(dgl_graph, edge_id)
            
            if not(self.omit_most_frequent_edge and 
                   self._dataset_loader.get_edge_type(dgl_graph, edge_id) == self.most_frequent_edge_type):  # 边未被忽略
                result_token_ids.extend(edge_tokens)
                if len(edge_tokens) > 1:
                    result_element_ids.append(f"START_EDGE_{edge_id}")
                    for j, token in enumerate(edge_tokens[1:-1]):
                        result_element_ids.append(f"edge_{edge_id}_dim_{j}")
                    result_element_ids.append(f"END_EDGE_{edge_id}")
                else:
                    result_element_ids.append(f"edge_{edge_id}")
       
        
        return result_token_ids, result_element_ids
      
    #===================== 统计信息收集 ======================
    
    
    def _collect_statistics_from_graphs(self, graph_data_list: List[Dict[str, Any]]) -> None:
        """收集全局统计信息（内部方法）
        
        - 仅保留数值三元组频率热路径；字符串统计可选（默认关闭）
        - 两阶段：先全局维度推断，再分片计数
        - 支持多进程分片（fork），汇总后一次性归一化
        - 不写死数据集假设，由数据层提供类型空间维度；若不可用则按实际 max id 推断
        """
        import torch as _torch

        N = len(graph_data_list)
        if N == 0:
            self._triplet_frequency_tensor = _torch.zeros((0, 0, 0), dtype=_torch.long)
            self._triplet_tensor_dims = (0, 0, 0)
            self.triplet_frequencies = {}
            self.two_hop_frequencies = {}
            self.statistics_collected = True
            return

        # 阶段1：全局类型空间维度推断（max id + 1）
        if self.stats_parallel_enabled:
            if mp.get_start_method(allow_none=True) not in ("fork", None):
                raise RuntimeError("统计并行需要 fork 平台（Linux/posix）")
            W = max(1, int(self.parallel_num))
            W = min(W, N)
            chunk_sizes = [(N // W) + (1 if i < (N % W) else 0) for i in range(W)]
            ranges = []
            s = 0
            for sz in chunk_sizes:
                if sz > 0:
                    ranges.append((s, s + sz))
                s += sz
            global _MP_STATS_SERIALIZER, _MP_STATS_GRAPH_LIST
            _MP_STATS_SERIALIZER = self
            _MP_STATS_GRAPH_LIST = graph_data_list
            max_v = 0
            max_e = 0
            with mp.Pool(processes=W, initializer=_mp_stats_init, initargs=(self, graph_data_list)) as pool:
                for mv, me in pool.imap_unordered(_mp_stats_max_range, ranges, chunksize=1):
                    if mv > max_v:
                        max_v = mv
                    if me > max_e:
                        max_e = me
            Vdim0 = int(max_v)
            Edim = int(max_e)
        else:
            Vdim0 = 0
            Edim = 0
            for i in range(N):
                g = graph_data_list[i].get('dgl_graph')
                if g is None:
                    continue
                node_type_ids = self._dataset_loader.get_graph_node_type_ids(g)
                edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(g)
                if node_type_ids.numel() > 0:
                    Vdim0 = max(Vdim0, int(node_type_ids.max().item()) + 1)
                if edge_type_ids.numel() > 0:
                    Edim = max(Edim, int(edge_type_ids.max().item()) + 1)

        if Vdim0 <= 0 or Edim <= 0:
            raise ValueError("无法推断类型空间维度。请检查数据集的类型 id 特征。")

        Vdim1 = Vdim0

        # 阶段2：计数（稀疏），不再构建致密三维张量
        sparse_counts: Dict[int, int] = {}
        if self.stats_parallel_enabled:
            if mp.get_start_method(allow_none=True) not in ("fork", None):
                raise RuntimeError("统计并行需要 fork 平台（Linux/posix）")
            W = max(1, int(self.parallel_num))
            W = min(W, N)
            chunk_sizes = [(N // W) + (1 if i < (N % W) else 0) for i in range(W)]
            ranges = []
            s = 0
            for sz in chunk_sizes:
                if sz > 0:
                    ranges.append((s, s + sz))
                s += sz
            # 这里不需要再次声明 global，直接复用上一段已声明的全局引用
            _MP_STATS_SERIALIZER = self
            _MP_STATS_GRAPH_LIST = graph_data_list
            with mp.Pool(processes=W, initializer=_mp_stats_init, initargs=(self, graph_data_list)) as pool:
                args_list = [(lo, hi, Vdim0, Edim) for (lo, hi) in ranges]
                for part in pool.imap_unordered(_mp_stats_count_range_sparse, args_list, chunksize=1):
                    for k, v in part.items():
                        sparse_counts[k] = sparse_counts.get(k, 0) + int(v)
        else:
            # 串行稀疏计数
            for i in range(N):
                g = graph_data_list[i].get('dgl_graph')
                if g is None:
                    continue
                if self.enable_string_stats:
                    _ = self._extract_all_statistics(g)  # 可选
                node_type_ids = self._dataset_loader.get_graph_node_type_ids(g)
                edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(g)
                if edge_type_ids.numel() == 0:
                    continue
                src, dst = self._get_all_edges_from_heterograph(g)
                src_t = node_type_ids.index_select(0, src).long()
                dst_t = node_type_ids.index_select(0, dst).long()
                et_t = edge_type_ids.long()
                if src_t.numel() == 0:
                    continue
                if src_t.max() >= Vdim0 or dst_t.max() >= Vdim1 or et_t.max() >= Edim:
                    raise ValueError("统计计数阶段出现类型ID越界，请检查维度推断逻辑。")
                lin = (src_t * (Edim * Vdim1) + et_t * Vdim1 + dst_t).to(_torch.long)
                if lin.numel() == 0:
                    continue
                lin_sorted, _ = _torch.sort(lin)
                uniq, cts = _torch.unique_consecutive(lin_sorted, return_counts=True)
                for k, v in zip(uniq.tolist(), cts.tolist()):
                    sparse_counts[k] = sparse_counts.get(k, 0) + int(v)

        # 保存维度与稀疏键值缓存；不再使用致密张量
        self._triplet_tensor_dims = (Vdim0, Edim, Vdim1)
        self._triplet_frequency_tensor = None
        if len(sparse_counts) == 0:
            self._triplet_sparse_counts = {}
            self._triplet_sparse_keys = None
            self._triplet_sparse_vals = None
        else:
            keys = _torch.tensor(sorted(sparse_counts.keys()), dtype=_torch.long)
            vals = _torch.tensor([sparse_counts[int(k)] for k in keys.tolist()], dtype=_torch.int32)
            self._triplet_sparse_counts = sparse_counts
            self._triplet_sparse_keys = keys
            self._triplet_sparse_vals = vals

        # 与旧接口对齐（字符串表为空；保留字段）
        self.triplet_frequencies = {}
        self.two_hop_frequencies = {}
        self.statistics_collected = True
        logger.info("✅ GraphSeq稀疏统计收集完成(并行=%s): V=%d, E=%d, 非零=%d, 总计数=%d",
                    str(self.stats_parallel_enabled), Vdim0, Edim, len(self._triplet_sparse_counts), int(sum(self._triplet_sparse_counts.values())))
        
    def _extract_all_statistics(self, dgl_graph: dgl.DGLGraph) -> Tuple[Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str, str, str], int]]:
        """一次性提取所有统计信息：三元组和两跳路径"""
        
        # 构建邻接表（一次构建，多次使用）
        adjacency = defaultdict(list)
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        # 使用线程本地映射，避免跨线程共享
        setattr(self._thread_local, 'current_edge_id_mapping', self._build_edge_id_mapping(dgl_graph))
        
        for i, (src, dst) in enumerate(zip(src_nodes.numpy(), dst_nodes.numpy())):
            src, dst = int(src), int(dst)
            edge_type = self._get_edge_type(dgl_graph, src, dst)
            adjacency[src].append((dst, edge_type))
            adjacency[dst].append((src, edge_type))  # 无向图，添加双向边
        
        # 收集三元组统计
        triples = defaultdict(int)
        for src in adjacency:
            for dst, edge_type in adjacency[src]:
                src_type, dst_type = self._get_node_type(dgl_graph, src), self._get_node_type(dgl_graph, dst)
                # 每条边贡献两个triples（双向）
                triples[(src_type, edge_type, dst_type)] += 1
                # triples[(dst_type, edge_type, src_type)] += 1
        
        # 收集两跳路径统计（目前按设计不启用两跳路径统计，仅保留占位注释以免误解）
        # two_hop_paths = defaultdict(int)
        #
        # for node1 in range(dgl_graph.num_nodes()):
        #     node1_type = self._get_node_type(dgl_graph, node1)
        #     # 遍历node1的邻居
        #     for node2, edge1_type in adjacency[node1]:
        #         node2_type = self._get_node_type(dgl_graph, node2)
        #         # 遍历node2的邻居，形成两跳路径
        #         for node3, edge2_type in adjacency[node2]:
        #             node3_type = self._get_node_type(dgl_graph, node3)
        #             # 记录两跳路径（如需启用，请确保确定性处理）
        #             path = (node1_type, edge1_type, node2_type, edge2_type, node3_type)
        #             two_hop_paths[path] += 1
        #
        # return dict(triples), dict(two_hop_paths)
        # 按当前设计，返回三元组统计，two_hop 为空字典
        return dict(triples), {}
    
    def _calculate_edge_weights(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], float]:
        """
        直接从DGL图计算边权重，优先使用稀疏三元组统计；如无则回退到致密张量；再无则回退到字符串频率表。
        """
        import torch as _torch
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        node_type_ids = self._dataset_loader.get_graph_node_type_ids(dgl_graph)
        edge_type_ids = self._dataset_loader.get_graph_edge_type_ids(dgl_graph)

        src_t = node_type_ids.index_select(0, src_nodes)
        dst_t = node_type_ids.index_select(0, dst_nodes)
        et_t = edge_type_ids

        # 1) 优先使用稀疏统计
        if self._triplet_sparse_keys is not None and self._triplet_sparse_vals is not None:
            V0, E0, V1 = self._triplet_tensor_dims
            if src_t.numel() == 0:
                return {}
            if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
                raise ValueError("类型ID超出频率统计的维度，请使用完整数据集进行统计构建")
            lin = (src_t.long() * (E0 * V1) + et_t.long() * V1 + dst_t.long())
            # searchsorted 查表
            pos = _torch.searchsorted(self._triplet_sparse_keys, lin)
            # 等值掩码（避免不存在键时误取）
            valid = (pos < self._triplet_sparse_keys.shape[0]) & (self._triplet_sparse_keys.index_select(0, pos) == lin)
            counts = _torch.zeros_like(lin, dtype=_torch.int32)
            if valid.any():
                counts[valid] = self._triplet_sparse_vals.index_select(0, pos[valid])
            weights = _torch.log10(counts.clamp(min=1).to(_torch.float32))
            return {(int(s), int(d)): float(w) for s, d, w in zip(src_nodes.tolist(), dst_nodes.tolist(), weights.tolist())}

        # 2) 次选致密张量
        if self._triplet_frequency_tensor is not None:
            V0, E0, V1 = self._triplet_tensor_dims
            if src_t.numel() == 0:
                return {}
            if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
                raise ValueError("类型ID超出频率张量的维度，请使用完整数据集进行统计构建")
            freq_vals = self._triplet_frequency_tensor[src_t.long(), et_t.long(), dst_t.long()].clamp_min_(1).to(_torch.float32)
            weights = _torch.log10(freq_vals)
            return {(int(s), int(d)): float(w) for s, d, w in zip(src_nodes.tolist(), dst_nodes.tolist(), weights.tolist())}

        # 3) 最后回退到字符串频率表（需要完备）
        edge_weights: Dict[Tuple[int, int], float] = {}
        for s, d, etid in zip(src_nodes.tolist(), dst_nodes.tolist(), et_t.tolist()):
            s_name = self._get_node_type(dgl_graph, int(s))
            d_name = self._get_node_type(dgl_graph, int(d))
            e_name = self._get_edge_type(dgl_graph, int(s), int(d))
            triplet = (s_name, e_name, d_name)
            if triplet not in self.triplet_frequencies:
                raise ValueError(f"统计缺失: 三元组{triplet}未在频率表中。请在初始化时传入更多图以收集充分统计。")
            edge_weights[(int(s), int(d))] = log10(self.triplet_frequencies[triplet])
        return edge_weights
        V0, E0, V1 = self._triplet_tensor_dims
        if src_t.numel() == 0:
            return {}
        if src_t.max().item() >= V0 or dst_t.max().item() >= V1 or et_t.max().item() >= E0:
            raise ValueError("类型ID超出频率张量的维度，请使用完整数据集进行统计构建")

        freq_vals = self._triplet_frequency_tensor[src_t.long(), et_t.long(), dst_t.long()].clamp_min_(1).to(_torch.float32)
        weights = _torch.log10(freq_vals)
        return {(int(s), int(d)): float(w) for s, d, w in zip(src_nodes.tolist(), dst_nodes.tolist(), weights.tolist())}
    
    def _split_connected_components(self, dgl_graph: dgl.DGLGraph) -> List[dgl.DGLGraph]:
        """
        将DGL图拆分为连通分量子图列表
        
        该函数将不连通的图拆分为多个连通的子图，每个子图保持原始节点ID不变。
        这对于需要连通图的序列化算法（如中国邮递员算法、欧拉回路算法）很有用。
        
        Args:
            dgl_graph: 输入的DGL图
            
        Returns:
            List[dgl.DGLGraph]: 连通分量子图列表，按连通分量大小降序排列
            
        Note:
            - 每个子图保持原始节点ID，便于后续token获取
            - 返回的子图按连通分量大小降序排列，便于处理
        """
        import networkx as nx
        
        # 转换为NetworkX图以使用其连通分量算法
        nx_graph = nx.Graph()
        
        # 添加所有节点
        for i in range(dgl_graph.num_nodes()):
            nx_graph.add_node(i)
        
        # 添加所有边
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            nx_graph.add_edge(src, dst)
        
        # 获取连通分量
        components = list(nx.connected_components(nx_graph))
        
        # 按连通分量大小降序排列
        components.sort(key=len, reverse=True)
        
        # 为每个连通分量创建子图
        subgraphs = []
        for component in components:
            if len(component) == 0:
                continue
                
            # 创建子图，DGL会自动重新映射节点ID
            # 由于dataloader有fallback逻辑，不需要保持原始节点ID
            subgraph = dgl.node_subgraph(dgl_graph, list(component))
            subgraphs.append(subgraph)
        
        return subgraphs
    
    def _convert_dgl_to_networkx(self, dgl_graph: dgl.DGLGraph) -> nx.MultiGraph:
        """将DGL图转换为NetworkX多重图（公共方法）
        
        DGL中的双向边(u,v)和(v,u)代表一条无向边，但NetworkX会将它们视为重复。
        我们需要去重，只保留每条无向边的一个方向。
        
        Args:
            dgl_graph: DGL图
            
        Returns:
            nx.MultiGraph: NetworkX多重图
        """
        
        G = nx.MultiGraph()
        
        # 添加节点
        G.add_nodes_from(range(dgl_graph.num_nodes()))
        
        # 获取边信息
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # 去重：对于每条无向边，只添加一次
        added_edges = set()
        edge_list = []
        
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            # 标准化边的表示（小节点号在前）
            edge_key = (min(src, dst), max(src, dst))
            
            if edge_key not in added_edges:
                added_edges.add(edge_key)
                edge_list.append((src, dst))
                G.add_edge(src, dst, weight=1)
        
        return G