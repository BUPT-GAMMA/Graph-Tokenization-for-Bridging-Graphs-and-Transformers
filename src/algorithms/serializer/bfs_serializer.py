"""
广度优先搜索序列化器
"""

from typing import Dict, Any, List, Tuple, Optional
import dgl
from collections import deque
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger
import networkx as nx

logger = get_logger(__name__)

class BFSSerializer(BaseGraphSerializer):
    """广度优先搜索序列化器"""
    
    def __init__(self):
        super().__init__()
        self.name = "bfs"
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        初始化BFS序列化器（统一接口实现）
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表（BFS不需要统计信息）
        """
        # 保存数据集加载器引用
        self._dataset_loader = dataset_loader
        
        # BFS序列化器不需要预处理统计信息
        self._dataset_stats.update({
            'method': 'BFS',
            'description': '广度优先搜索序列化',
            'requires_statistics': False
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
        
        # 调用新的序列化逻辑，返回token和element序列
        token_sequence, element_sequence = self._BFS_serialize(graph_data, start_node=start_node)
        
        # 构建SerializationResult
        dgl_graph = graph_data['dgl_graph']
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _BFS_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """
        使用BFS序列化图，返回token序列和element序列
        
        Args:
            mol_data: 图数据，包含dgl_graph
            dataset_loader: 数据集加载器，用于获取节点token表示
            start_node: 起始节点索引
            
        Returns:
            Tuple[List[int], List[str]]: (token序列, element序列)
        """
        # 使用公共方法验证图数据
        dgl_graph  = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        # 单节点特殊情况
        if num_nodes == 1:
            return [self.get_node_token(dgl_graph, 0)], ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "❌ 起始节点索引超出范围"
        
        # 构建邻接列表
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        
        # 执行BFS
        visited = [False] * num_nodes
        result_token_ids = []
        result_element_ids = []
        queue = deque()
        
        # 从起始节点开始BFS
        queue.append(start_node)
        visited[start_node] = True
        
        while queue:
            node = queue.popleft()
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
            
            # 按索引排序邻居确保确定性
            neighbors = sorted(adj_list[node])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        assert len(result_token_ids) == len(result_element_ids)
        # assert visited == [True] * num_nodes # 有些图不连通，visited不全为True
        
        return result_token_ids, result_element_ids
    