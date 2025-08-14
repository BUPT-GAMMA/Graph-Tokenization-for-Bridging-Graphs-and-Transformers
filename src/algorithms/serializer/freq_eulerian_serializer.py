"""
GraphSeq序列化器 - 简化版本
===========================

基于频率引导的欧拉回路序列化算法，直接使用DGL图进行权重计算。
移除了不必要的预处理步骤，简化了数据流。

说明：当前仅使用“三元组频率”作为引导信号；两跳路径统计保留为注释占位，属于预期不启用的行为。
"""

from typing import Dict, Any, List, Optional, Tuple
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from collections import defaultdict
import dgl
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger
import networkx as nx

# 设置logger
logger = get_logger(__name__)


class FeulerSerializer(BaseGraphSerializer):
    """GraphSeq序列化器 - 基于频率引导的欧拉回路，直接使用DGL图"""
    
    def __init__(self, verbose: bool = False, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True):
        super().__init__()
        self.name = "feuler"
        self.triplet_frequencies = defaultdict(int)  # 三元组频率
        self.two_hop_frequencies = defaultdict(int)  # 两跳路径频率
        self.statistics_collected = False
        self.verbose = verbose  # 是否输出详细日志
        self.include_edge_tokens = include_edge_tokens  # 控制是否包含边token
        self.omit_most_frequent_edge = omit_most_frequent_edge  # 控制是否省略最高频边类型
        
        # 添加运行时统计
        self.serialization_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'avg_tokens_per_molecule': 0.0
        }
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        初始化GraphSeq序列化器（统一接口实现）
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表，用于统计信息收集
        """
        # 如果有图数据列表，收集统计信息
        assert graph_data_list is not None, "❌ 图数据列表不能为空"
        logger.info(f"📊 GraphSeq开始收集全局统计信息，共 {len(graph_data_list)} 个图")
        self._collect_statistics_from_graphs(graph_data_list)
        
        # 更新数据集统计信息
        self._dataset_stats.update({
            'statistics_collected': self.statistics_collected,
            'triplet_frequencies_count': len(self.triplet_frequencies),
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'two_hop_frequencies_enabled': False,
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        序列化单个图（统一接口实现）
        
        Args:
            graph_data: 图数据，包含dgl_graph等字段
            **kwargs: 额外的序列化参数
            
        Returns:
            SerializationResult: 序列化结果
        """
        # 获取起始节点参数
        start_node = kwargs.get('start_node', 0)
        
        dgl_graph = self._validate_graph_data(graph_data)
        
        # 检查连通性
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # 连通图，使用原有逻辑
            token_sequence, element_sequence = self._gseq_serialize(graph_data, start_node)
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([token_sequence], [element_sequence], id_mapping)
        else:
            self._current_edge_id_mapping=None
            # 不连通图，分别处理每个连通分量
            logger.debug("⚠️ 图不连通，将分别处理各个连通分量")
            
            # 拆分连通分量
            subgraphs = self._split_connected_components(dgl_graph)
            logger.debug(f"📊 找到 {len(subgraphs)} 个连通分量")
            
            # 分别处理每个连通分量
            all_token_sequences = []
            all_element_sequences = []
            
            for i, subgraph in enumerate(subgraphs):
                logger.debug(f"🔄 处理第 {i+1}/{len(subgraphs)} 个连通分量，节点数: {subgraph.num_nodes()}")
                
                # 为子图创建新的图数据
                subgraph_data = {
                    'dgl_graph': subgraph,
                    **{k: v for k, v in graph_data.items() if k != 'dgl_graph'}
                }
                
                # 处理子图（使用子图中的第一个节点作为起始节点）
                sub_start_node = 0  # 子图中的节点ID从0开始
                token_seq, element_seq = self._gseq_serialize(subgraph_data, sub_start_node)
                
                all_token_sequences.append(token_seq)
                all_element_sequences.append(element_seq)
            
            # 合并所有序列
            merged_token_sequence = []
            merged_element_sequence = []
            
            for i, (token_seq, element_seq) in enumerate(zip(all_token_sequences, all_element_sequences)):
                if i > 0:
                    logger.debug(f"🔗 连接第 {i} 个连通分量的序列")
                    
                    # 添加分隔符token
                    merged_token_sequence.append(self._dataset_loader.config.component_sep_token_id)
                    merged_element_sequence.append("component_sep")
                
                merged_token_sequence.extend(token_seq)
                merged_element_sequence.extend(element_seq)
            
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([merged_token_sequence], [merged_element_sequence], id_mapping)
    
    def _gseq_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """
        使用频率引导的欧拉回路序列化图，返回token序列和element序列
        
        Args:
            mol_data: 图数据，包含dgl_graph
            start_node: 起始节点索引（GraphSeq有自己的选择策略）
            
        Returns:
            Tuple[List[int], List[str]]: (token序列, element序列)
        """
        # 使用公共方法验证图数据
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
                
        # 单节点特殊情况
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "❌ 起始节点索引超出范围"
        
        # 直接查找频率引导的欧拉回路，不需要预处理
        eulerian_path = self._find_frequency_guided_eulerian_circuit(dgl_graph, start_node)
        
        if not eulerian_path:
            raise ValueError("❌ 无法找到频率引导的欧拉回路")
        
        return self._convert_path_to_tokens(eulerian_path, mol_data)
    
    def _find_frequency_guided_eulerian_circuit(self, dgl_graph: dgl.DGLGraph, start_node: int = 0) -> Optional[List[int]]:
        """
        直接使用DGL图查找频率引导的欧拉回路
        
        Args:
            dgl_graph: DGL图
            start_node: 起始节点
            
        Returns:
            Optional[List[int]]: 欧拉回路路径
        """
        # 直接计算每条边的频率权重
        # 计算边权时，缺失的三元组频率可能导致 KeyError。按 C2 文档：不引入回退逻辑，但这里改为明确报错，指示初始化统计不足。
        edge_weights = self._calculate_edge_weights(dgl_graph)
        
        # 构建带权重的邻接表
        weighted_graph = self._build_weighted_adjacency_list(dgl_graph, edge_weights)
        
        # 确保起始节点有出边
        if start_node not in weighted_graph or not weighted_graph[start_node]:
            # 找到第一个有出边的节点作为起始点
            for i in range(dgl_graph.num_nodes()):
                if weighted_graph[i]:
                    start_node = i
                    break
            else:
                raise ValueError("❌ 没有节点有出边")
        
        logger.debug(f"🔍 从节点{start_node}开始查找频率引导的欧拉回路...")
        
        # 使用频率引导的Hierholzer算法
        return self._frequency_guided_hierholzer(weighted_graph, start_node)
    
    def _build_weighted_adjacency_list(self, dgl_graph: dgl.DGLGraph, 
                                     edge_weights: Dict[Tuple[int, int], float]) -> Dict[int, List[Tuple[int, float]]]:
        """
        构建带权重的邻接表
        
        Args:
            dgl_graph: DGL图
            edge_weights: 边权重字典
            
        Returns:
            Dict[int, List[Tuple[int, float]]]: 带权重的邻接表
        """
        weighted_graph = defaultdict(list)
        
        # 获取所有边
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # 构建邻接表
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            weight = edge_weights[(src, dst)]
            weighted_graph[src].append((dst, weight))
        
        # 按权重降序、邻居ID升序排序（同权时按ID稳定tie-breaker）
        for node in weighted_graph:
            weighted_graph[node].sort(key=lambda x: (-x[1], x[0]))
        
        return weighted_graph
    
    def _frequency_guided_hierholzer(self, weighted_graph: Dict[int, List[Tuple[int, float]]], 
                                   start_node: int) -> List[int]:
        """
        频率引导的Hierholzer算法
        
        在每个节点选择下一条边时，优先选择高频率的边
        
        Args:
            weighted_graph: 带权重的邻接表
            start_node: 起始节点
            
        Returns:
            List[int]: 欧拉回路路径
        """
        # 复制图结构（避免修改原数据）
        graph = defaultdict(list)
        for node, neighbors in weighted_graph.items():
            graph[node] = neighbors.copy()
        
        # Hierholzer算法主循环
        current_path = [start_node]
        circuit = []
        current_node = start_node
        
        while current_path:
            if graph[current_node]:
                # 选择权重最高（tie时最小ID）的边（列表已按(-weight, neighbor_id)排序）
                next_node, weight = graph[current_node].pop(0)
                current_path.append(next_node)
                current_node = next_node
            else:
                # 如果当前节点没有未访问的出边，将其加入回路
                circuit.append(current_path.pop())
                if current_path:
                    current_node = current_path[-1]
        
        # 反转得到正确的路径顺序
        circuit.reverse()
        
        logger.debug(f"✅ 找到频率引导的欧拉回路: 长度={len(circuit)}")
        return circuit

    