"""
Unified Data Interface
======================

设计原则：
- 单一数据文件 + 索引划分；
- 对上层（序列化/BPE/训练）提供统一读取接口；
- 序列化与 BPE 的缓存管理是接口内部细节，数据构建需显式调用；
- 简化接口：两个核心方法 get_sequences() 和 get_sequences_by_splits()。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.data.base_loader import BaseDataLoader
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.utils.logger import get_logger

# from src.models.bert.vocab_manager import build_vocab_from_sequences  # 延迟导入避免循环依赖
logger = get_logger(__name__)

@dataclass
class UnifiedDataInterface:
    config: ProjectConfig
    dataset: str
    # 运行期可选的预加载缓存（同一进程内复用，避免重复IO/构建）
    _preloaded_graphs: List[Dict[str, Any]] | None = None
    _preloaded_splits: Dict[str, List[int]] | None = None
    _loader: BaseDataLoader | None = None

    def _resolve_processed_dir(self) -> Path:
        return Path(self.config.processed_data_dir) / self.dataset


    def _load_split_indices(self) -> Dict[str, List[int]]:
        """加载原有格式的划分索引文件"""
        # 使用项目配置的 data_dir，确保加载真实项目数据而非当前工作目录
        data_dir = Path(self.config.data_dir) / self.dataset
        
        train_path = data_dir / "train_index.json"
        val_path = data_dir / "val_index.json"
        test_path = data_dir / "test_index.json"
        
        splits = {}
        
        # 加载训练集索引
        if train_path.exists():
            with open(train_path, 'r') as f:
                splits['train'] = json.load(f)
        else:
            raise FileNotFoundError(f"训练集索引文件不存在: {train_path}")
        
        # 加载验证集索引  
        if val_path.exists():
            with open(val_path, 'r') as f:
                splits['val'] = json.load(f)
        else:
            raise FileNotFoundError(f"验证集索引文件不存在: {val_path}")
        
        # 加载测试集索引
        if test_path.exists():
            with open(test_path, 'r') as f:
                splits['test'] = json.load(f)
        else:
            raise FileNotFoundError(f"测试集索引文件不存在: {test_path}")
        
        return splits

    # ----------------------- 预加载图/索引（避免重复构建） -----------------------
    def preload_graphs(self) -> None:
        """主动加载并缓存全量图与划分索引，供后续序列化/BPE 构建复用。"""
        loader = self.get_dataset_loader()
        graphs, _ = loader.get_all_data_with_indices()
        self._loader=loader
        self._preloaded_graphs = graphs
        self._preloaded_splits = self._load_split_indices()

    def _get_serialization_cache_key(self) -> str:
        """生成基于配置的序列化缓存键"""
        # 从新的配置路径读取多重采样设置
        ms = self.config.serialization.multiple_sampling
        use_multi = getattr(ms, 'enabled', False)
        num_realizations = getattr(ms, 'num_realizations', 1)
        
        if use_multi and num_realizations > 1:
            return f"multi_{num_realizations}"
        else:
            return "single"
    
    def _load_serialization_result(self, method: str) -> Dict[str, Any]:
        """内部方法：加载序列化结果"""
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        result_path = base / "serialized_data" / method / cache_key / "serialized_data.pickle"
        
        if not result_path.exists():
            raise FileNotFoundError(f"序列化结果不存在: {result_path}")
        
        with open(result_path, 'rb') as f:
            data = pickle.load(f)
            
        assert 'serialization_method' in data, "序列化结果文件缺少 'serialization_method' 字段"
        assert data['serialization_method'] == method, f"请求的序列化方法 '{method}' 与保存的方法 '{data['serialization_method']}' 不匹配"
        
        return data

    # ----------------------- 构建与持久化（显式触发） -----------------------
    def _extract_properties_from_graphs(self, graphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从原始图数据中提取属性信息（数值/短字符串），用于写入缓存。

        说明：保持与 data_prepare 中的提取口径一致，避免依赖底层 data.pkl。
        """
        try:
            import numpy as np  # 局部导入，避免顶层依赖
        except Exception:
            np = None  # 测试环境可不严格依赖

        graph_structure_keys = {
            'dgl_graph', 'graph', 'edge_index', 'edge_attr', 'node_features', 'edge_features',
            'num_nodes', 'num_edges', 'adjacency', 'adjacency_matrix', 'node_attr',
            'smiles', 'mol', 'molecule', 'rdkit_mol',
            'id', 'index', 'idx', 'dataset_name', 'data_type',
            'node_ids', 'edge_ids', 'global_node_ids',
            'smiles_1', 'smiles_2', 'smiles_3', 'smiles_4'
        }

        properties: List[Dict[str, Any]] = []
        for item in graphs:
            prop: Dict[str, Any] = {}
            if isinstance(item, dict):
                if 'properties' in item and isinstance(item['properties'], dict):
                    prop.update(item['properties'])
                for k, v in item.items():
                    if k in graph_structure_keys:
                        continue
                    if isinstance(v, (int, float)):
                        prop[k] = float(v)
                    elif isinstance(v, str) and len(v) < 50:
                        prop[k] = v
                    elif np is not None and isinstance(v, (np.integer, np.floating)):
                        prop[k] = float(v)
            properties.append(prop)
        return properties

    def _build_and_persist_serialization(self, method: str) -> Path:
        """在缓存缺失时，按当前配置确定性地构建序列化结果并持久化，返回结果路径。"""
        # 获取数据
        if self._preloaded_graphs is not None:
            graphs = self._preloaded_graphs
            loader = self.get_dataset_loader()
        else:
            loader = self.get_dataset_loader()
            graphs, _ = loader.get_all_data_with_indices()

        # 初始化序列化器
        serializer = SerializerFactory.create_serializer(method)
        serializer.initialize_with_dataset(loader, graphs)

        # multiple 逻辑：按配置决定是否为每图产生多变体
        # 从新的配置路径读取多重采样设置
        ms = self.config.serialization.multiple_sampling
        use_multi = bool(getattr(ms, 'enabled', False))
        num_realizations = int(getattr(ms, 'num_realizations', 1))

        sequences: List[List[int]] = []
        graph_ids: List[int] = []
        flattened_properties: List[Dict[str, Any]] = []

        # 提取属性（用于与序列对齐）
        properties = self._extract_properties_from_graphs(graphs)

        if use_multi and num_realizations > 1:
            # 启用内部多进程并行（fork-only），并使用 CPU 核心数作为 workers
            batch_results = serializer.batch_multiple_serialize(
                graphs,
                num_samples=num_realizations,
                desc=f"serialize-multi-{method}",
                parallel=True,
            )
            # 逐图展开变体
            for gid, res in enumerate(batch_results):
                if res is None:
                    raise ValueError(f"序列化失败: 图 {gid} 返回空结果")
                if not hasattr(res, 'token_sequences'):
                    raise ValueError(f"序列化结果格式错误: 图 {gid} 缺少 token_sequences 属性")
                if not res.token_sequences:
                    raise ValueError(f"序列化结果为空: 图 {gid} 的 token_sequences 为空")
                    
                for vid, seq in enumerate(res.token_sequences):
                    sequences.append(seq)
                    graph_ids.append(gid)
                    assert gid < len(properties), f"属性索引越界: 图 {gid} 超出属性列表长度 {len(properties)}"
                    flattened_properties.append(properties[gid])
        else:
            # 单次序列化
            batch_results = serializer.batch_serialize(graphs, desc=f"serialize-{method}")
            for gid, res in enumerate(batch_results):
                if res is None:
                    raise ValueError(f"序列化失败: 图 {gid} 返回空结果")
                if not hasattr(res, 'token_sequences'):
                    raise ValueError(f"序列化结果格式错误: 图 {gid} 缺少 token_sequences 属性")
                if not res.token_sequences:
                    raise ValueError(f"序列化结果为空: 图 {gid} 的 token_sequences 为空")
                    
                sequences.append(res.token_sequences[0])
                graph_ids.append(gid)
                if gid >= len(properties):
                    raise IndexError(f"属性索引越界: 图 {gid} 超出属性列表长度 {len(properties)}")
                flattened_properties.append(properties[gid])

        # 写入
        cache_key = self._get_serialization_cache_key()
        out_dir = self._resolve_processed_dir() / "serialized_data" / method / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "serialized_data.pickle"
        with out_path.open('wb') as f:
            pickle.dump({
                'sequences': sequences,
                'properties': flattened_properties,
                'serialization_method': method,
                'graph_ids': graph_ids,
            }, f)
        return out_path



    # ----------------------- BPE codebook 与 Transform -----------------------
    def get_bpe_codebook(self, method: str) -> Dict[str, Any]:
        """读取 BPE codebook（按 single/multi_<k> 分目录管理）。"""
        cache_key = self._get_serialization_cache_key()
        model_path = self.config.model_dir / "bpe" / self.dataset / method / cache_key / "bpe_codebook.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"BPE codebook 不存在: {model_path}")
            
        with model_path.open('rb') as f:
            data = pickle.load(f)
            
        assert isinstance(data, dict) and 'merge_rules' in data and 'vocab_size' in data, "BPE codebook 格式错误：缺少 merge_rules 或 vocab_size"
            
        return {'merge_rules': data['merge_rules'], 'vocab_size': int(data['vocab_size'])}

    def get_bpe_encoder(self, method: str, *, encode_backend: str = 'cpp', **engine_kwargs):
        """读取 codebook 并返回可直接用于编码的 BPEEngine。

        说明：
        - 路径与持久化仍由 UDI 管理（按 dataset/method 分层）
        - 此方法仅做“读取 -> 构造编码引擎”的桥接
        - encode_backend 默认为 'cpp'，需已构建 C++ 扩展
        """
        from src.algorithms.compression.bpe_engine import BPEEngine  # 局部导入避免循环依赖
        codebook = self.get_bpe_codebook(method)
        engine = BPEEngine.from_codebook_dict(codebook, encode_backend=encode_backend, **engine_kwargs)
        engine.build_encoder()
        return engine
    
    def save_bpe_codebook(self, method: str, merge_rules: List, vocab_size: int) -> Path:
        """保存 BPE codebook（按 single/multi_<k> 分目录）。
        
        Args:
            method: 序列化方法名
            merge_rules: BPE 合并规则列表
            vocab_size: 词汇表大小
            
        Returns:
            保存的文件路径
        """
        cache_key = self._get_serialization_cache_key()
        model_path = self.config.model_dir / "bpe" / self.dataset / method / cache_key / "bpe_codebook.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'merge_rules': merge_rules,
            'vocab_size': int(vocab_size)
        }
        
        with model_path.open('wb') as f:
            pickle.dump(data, f)
            
        return model_path



    # ----------------------- 核心序列读取接口 -----------------------
    def get_sequences(self, method: str) -> Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]]:
        """
        获取所有序列数据和标签
        
        Args:
            method: 序列化方法名
            
        Returns:
            Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]]:
                - 第一个元素: [(图ID, 序列), ...] 列表
                - 第二个元素: [标签属性字典, ...] 列表
        """
        serialized = self._load_serialization_result(method)
        
        assert 'sequences' in serialized, "序列化结果缺少必需字段 'sequences'"
        sequences = serialized['sequences']
        assert sequences, "序列化序列为空"
        
        assert 'graph_ids' in serialized, "序列化结果缺少必需字段 'graph_ids'"
        graph_ids = serialized['graph_ids']
            
        assert 'properties' in serialized, "序列化结果缺少必需字段 'properties'"
        properties = serialized['properties']
        if not properties:
            # 如果没有属性，创建空字典列表
            properties = [{} for _ in sequences]
            
        # 组装返回格式：图ID在前
        sequences_with_ids = [(gid, seq) for seq, gid in zip(sequences, graph_ids)]
        
        return sequences_with_ids, properties

    def get_sequences_by_splits(self, method: str) -> Tuple[
        List[Tuple[int, List[int]]], List[Dict[str, Any]],  # train
        List[Tuple[int, List[int]]], List[Dict[str, Any]],  # val  
        List[Tuple[int, List[int]]], List[Dict[str, Any]]   # test
    ]:
        """
        按训练/验证/测试划分获取序列数据
        
        Args:
            method: 序列化方法名
            
        Returns:
            Tuple of 6 elements:
                (train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels)
        """
        # 获取全部数据
        all_sequences, all_labels = self.get_sequences(method)
        
        # 获取划分索引
        split_indices = self._load_split_indices()
        
        # 构建图ID到索引的映射
        graph_id_to_indices = {}
        for idx, (graph_id, seq) in enumerate(all_sequences):
            if graph_id not in graph_id_to_indices:
                graph_id_to_indices[graph_id] = []
            graph_id_to_indices[graph_id].append(idx)
        
        def extract_split_data(split_name: str):
            if split_name not in split_indices:
                raise ValueError(f"无效的划分名称: {split_name}")
            
            split_graph_ids = set(split_indices[split_name])
            split_sequences = []
            split_labels = []
            
            for graph_id in split_graph_ids:
                if graph_id in graph_id_to_indices:
                    for idx in graph_id_to_indices[graph_id]:
                        split_sequences.append(all_sequences[idx])
                        split_labels.append(all_labels[idx])
            
            return split_sequences, split_labels
        
        train_seqs, train_labels = extract_split_data('train')
        val_seqs, val_labels = extract_split_data('val') 
        test_seqs, test_labels = extract_split_data('test')
        
        return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels

    def get_training_data(
        self,
        method: str,
    ) -> Tuple[
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # train (seqs_with_id, props)
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # val (seqs_with_id, props)
        Tuple[List[Tuple[int, List[int]]], List[Dict[str, Any]]],  # test (seqs_with_id, props)
    ]:
        """
        获取下游任务所需数据，为所有划分返回带graph_id的序列和完整的属性字典。
        """
        # get_sequences_by_splits 返回6个值，需要重新组织
        train_seqs, train_props, val_seqs, val_props, test_seqs, test_props = self.get_sequences_by_splits(method)
        train_data = (train_seqs, train_props)
        val_data = (val_seqs, val_props)
        test_data = (test_seqs, test_props)
        return train_data, val_data, test_data

    def get_training_data_flat(
        self,
        method: str,
    ) -> Tuple[
        List[List[int]], List[List[int]], List[List[int]]  # train, val, test sequences (flattened)
    ]:
        """
        获取扁平化的训练序列数据，专门用于预训练等不需要graph_id的场景。
        """
        (
            (train_seqs_with_id, _),
            (val_seqs_with_id, _),
            (test_seqs_with_id, _),
        ) = self.get_training_data(method)

        # 提取纯序列，丢弃graph_id
        train_sequences = [seq for _, seq in train_seqs_with_id]
        val_sequences = [seq for _, seq in val_seqs_with_id]
        test_sequences = [seq for _, seq in test_seqs_with_id]

        return train_sequences, val_sequences, test_sequences

    def get_training_data_flat_with_ids(
        self,
        method: str,
    ) -> Tuple[
        Tuple[List[List[int]], List[int]],  # train (sequences, graph_ids)
        Tuple[List[List[int]], List[int]],  # val (sequences, graph_ids)
        Tuple[List[List[int]], List[int]],  # test (sequences, graph_ids)
    ]:
        """
        获取扁平化的训练序列数据，包含graph_ids，用于预训练图级采样。
        """
        (
            (train_seqs_with_id, _),
            (val_seqs_with_id, _),
            (test_seqs_with_id, _),
        ) = self.get_training_data(method)

        # 提取序列和graph_id
        train_sequences = [seq for _, seq in train_seqs_with_id]
        train_gids = [gid for gid, _ in train_seqs_with_id]

        val_sequences = [seq for _, seq in val_seqs_with_id]
        val_gids = [gid for gid, _ in val_seqs_with_id]

        test_sequences = [seq for _, seq in test_seqs_with_id]
        test_gids = [gid for gid, _ in test_seqs_with_id]

        return (train_sequences, train_gids), (val_sequences, val_gids), (test_sequences, test_gids)

    def _resolve_target_property(self, requested_target_property: str | None) -> str | None:
        """
        [内部] 解析并确定最终要使用的target_property。
        """
        metadata = self.get_downstream_metadata()
        if 'downstream_label_keys' not in metadata:
            raise ValueError("数据集元数据缺少 'downstream_label_keys' 字段")
        available_labels = metadata['downstream_label_keys']
        
        if requested_target_property:
            if requested_target_property not in available_labels:
                raise ValueError(
                    f"请求的 target_property '{requested_target_property}' 不在可用属性列表中: {available_labels}"
                )
            return requested_target_property
        
        # 用户未指定，开始自动推断
        if not available_labels:
            # 对于像ZINC这样的数据集，可能没有显式的键，但有单一值，所以不会进入这个分支好
            # 这种隐式约定需要在数据加载器层处理，这里假设如果没有key就返回None
            return None
            
        if len(available_labels) == 1:
            # ！！！如果只有一个可用标签，自动选择它
            # 如果只有一个可用标签，自动选择它
            return available_labels[0]
        
        # 有多个标签，尝试使用默认值
        default_property = metadata.get('default_target_property')  # 这个字段可以是可选的
        if default_property and default_property in available_labels:
            return default_property
            
        # 无法解决歧义
        raise ValueError(
            f"数据集 '{self.dataset}' 有多个可用属性 {available_labels}，且无默认值。"
            f"请通过 target_property 参数明确指定一个。"
        )

    def _extract_labels_from_properties(self, properties: List[Dict[str, Any]], target_property: str) -> List[Any]:
        """
        [内部] 从属性字典列表中提取指定的目标标签。
        
        Args:
            properties: 属性字典列表
            target_property: 目标属性名，必须提供
        
        Returns:
            提取出的标签列表
        """
        labels = []
        for prop in properties:
            if target_property not in prop:
                raise ValueError(f"指定的target_property '{target_property}' 不在属性字典 {prop} 中。")
            labels.append(prop[target_property])
        return labels


    # ----------------------- 状态查询/报告 与 注册接口 -----------------------
    def has_serialized(self, method: str) -> bool:
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        result_path = base / "serialized_data" / method / cache_key / "serialized_data.pickle"
        return result_path.exists()

    def has_vocab(self, method: str, bpe: bool) -> bool:
        base = self._resolve_processed_dir()
        sub = "bpe" if bpe else "raw"
        vocab_dir = base / "vocab" / method / sub
        return (vocab_dir / "vocab.json").exists()

    def get_vocab(self, method: str):
        """读取与数据集绑定的词表。缺失直接报错。
        
        注意：词表与数据集绑定，包含原始token + BPE合并token + 特殊token的完整词表。
        无论运行时是否使用BPE，都从同一个完整词表中读取。
        """
        base = self._resolve_processed_dir()
        cache_key = self._get_serialization_cache_key()
        # 读取完整词表（包含所有原始token和BPE合并token），按 single/multi_<k> 分目录
        vocab_path = base / "vocab" / method / "bpe" / cache_key / "vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"词表不存在: {vocab_path}")
        from src.models.bert.vocab_manager import VocabManager  # 局部导入避免循环依赖
        return VocabManager.load_vocab(str(vocab_path), self.config)

    def register_vocab(self, vocab_manager, method: str) -> Path:
        """注册（落盘）词表到数据集的 processed 目录。

        注意：始终注册到完整词表位置，包含原始token + BPE合并token + 特殊token。
        
        返回保存路径。
        """
        base = self._resolve_processed_dir()
        # 注册到完整词表位置（包含所有token类型），按 single/multi_<k> 分目录
        cache_key = self._get_serialization_cache_key()
        out_dir = base / "vocab" / method / "bpe" / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "vocab.json"
        vocab_manager.save_vocab(str(out_path))
        return out_path

    def register_serialized_sequences(
        self,
        method: str,
        sequences: List[List[int]],
        properties: List[Dict[str, Any]] | None,
        split_indices: Dict[str, List[int]] | None,
    ) -> Path:
        """注册（落盘）外部提供的序列化结果。

        说明：split_indices 参数目前不单独落盘（索引文件位于 data_dir/<dataset>/ 下）。
        """
        cache_key = self._get_serialization_cache_key()
        out_dir = self._resolve_processed_dir() / "serialized_data" / method / cache_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "serialized_data.pickle"
        with out_path.open('wb') as f:
            pickle.dump({
                'sequences': sequences,
                'properties': properties or [],
                'serialization_method': method,
            }, f)
        return out_path

    def prepare_serialization(self, method: str) -> Path:
        """显式构建并持久化序列化结果（供顶层准备器使用）。"""
        return self._build_and_persist_serialization(method)

    def get_graphs(self) -> List[Dict[str, Any]]:
        # 读取图与属性（通过 loader），不在此方法内做切分
        if self._preloaded_graphs is not None:
            return self._preloaded_graphs
        loader = get_dataloader(self.dataset, self.config)
        all_data, _ = loader.get_all_data_with_indices()
        return all_data







    # 2) 纯内存端到端处理：已弃用（遵循“无隐式回退/不在 UDI 内构建”的规范）





    def get_split_indices(
        self,
    ) -> Dict[str, List[int]]:
        """
        获取数据集划分索引。
        
        Args:
            
        Returns:
            包含 'train', 'val', 'test' 索引列表的字典
        """
        # 直接加载原有的三个索引文件
        return self._load_split_indices()

    def get_dataset_loader(self) -> BaseDataLoader:
        """
        获取数据集加载器实例（仅供序列化器等内部组件使用）。
        
        Returns:
            BaseDataLoader 实例
        """
        if self._loader is not None:
            return self._loader
        else:
            self._loader = get_dataloader(self.dataset, self.config)
            return self._loader
    
    def get_num_classes(self) -> int:
        loader = self.get_dataset_loader()
        return loader.get_num_classes()
    
    def get_dataset_task_type(self) -> str:
        """获取数据集任务类型"""
        loader = self.get_dataset_loader()
        return loader.get_dataset_task_type()

    def get_loss_config(self) -> Optional[Dict[str, Any]]:
        """获取损失配置，支持超参覆盖"""
        # 1. 检查配置文件是否有覆盖
        if hasattr(self.config, 'task') and hasattr(self.config.task, 'loss_config') and self.config.task.loss_config:
            # 超参搜索覆盖
            return self.config.task.loss_config

        # 2. 返回数据集默认配置
        loader = self.get_dataset_loader()
        return loader.get_loss_config()

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """获取类别权重（代理DataLoader的方法）"""
        loader = self.get_dataset_loader()
        return loader.get_class_weights()

    def create_loss_function(self, task_type: str, num_classes: int) -> nn.Module:
        """
        根据配置创建损失函数

        Args:
            task_type: 任务类型
            num_classes: 类别数

        Returns:
            损失函数实例
        """
        loss_config = self.get_loss_config()

        if task_type == "mlm":
            return nn.CrossEntropyLoss(ignore_index=-100)
        elif task_type in ["binary_classification", "classification"]:
            if loss_config and loss_config.get('method') == 'focal':
                return self._create_focal_loss(loss_config)
            elif loss_config and loss_config.get('method') == 'weighted':
                return self._create_weighted_loss(loss_config, num_classes)
            else:
                return nn.CrossEntropyLoss()
        elif task_type == "regression":
            return nn.MSELoss()
        elif task_type == "multi_target_regression":
            return nn.L1Loss()
        elif task_type == "multi_label_classification":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    def _create_focal_loss(self, config: Dict[str, Any]) -> nn.Module:
        """创建Focal Loss（PyTorch原生实现）"""
        gamma = config.get('gamma', 2.0)
        alpha = config.get('alpha', 1.0)

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=1.0):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        return FocalLoss(gamma=gamma, alpha=alpha)

    def _create_weighted_loss(self, config: Dict[str, Any], num_classes: int) -> nn.Module:
        """创建加权交叉熵损失"""
        auto_weights = config.get('auto_weights', True)

        if auto_weights:
            # 从DataLoader获取自动计算的权重
            weights = self.get_class_weights()
            if weights is None:
                # 如果无法获取权重，使用均匀权重
                logger.warning("⚠️  无法计算类别权重，使用均匀权重")
                weights = torch.ones(num_classes)
        else:
            # 使用自定义权重
            custom_weights = config.get('weights')
            if custom_weights is not None:
                weights = torch.tensor(custom_weights, dtype=torch.float)
            else:
                weights = torch.ones(num_classes)

        return nn.CrossEntropyLoss(weight=weights)
    
    def create_empty_dataset_loader(self) -> BaseDataLoader:
        """
        创建“空”的数据集加载器实例：
        - 实际返回数据层中的 loader 实例；
        - 不触发任何数据加载/预处理动作；
        - 上层可使用该实例的元信息/约定方法。
        """
        return get_dataloader(self.dataset, self.config)









    # ----------------------- 下游任务元信息与空DataLoader -----------------------
    def get_downstream_metadata(self) -> Dict[str, Any]:
        """
        返回与下游任务相关的只读元信息：
        - label_keys
        - num_classes（若分类）
        - default_target_property（若回归）
        - dataset_name 等
        说明：不同数据集在对应 loader 内部硬编码其下游任务约定，UDI 汇总暴露。
        注意：label_shapes 在当前训练/评估管线中未使用，故不再强制要求。
        """
        loader = self.get_dataset_loader()
        meta: Dict[str, Any] = {
            'dataset': self.dataset,
        }
        # 直接假设所有loader都实现了这些方法，如未实现应在各自loader中补全
        for attr in [
            'get_downstream_label_keys',
            'get_num_classes',
            'get_default_target_property',
            'get_dataset_task_type',
        ]:
            # 基类已提供默认实现，子类可覆盖；此处直接调用
            meta[attr.replace('get_', '')] = getattr(loader, attr)()
        return meta
