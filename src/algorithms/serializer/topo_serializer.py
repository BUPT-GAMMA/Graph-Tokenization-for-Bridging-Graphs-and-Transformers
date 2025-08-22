"""
拓扑排序序列化器
==============

基于图的拓扑排序进行序列化，适用于有向无环图(DAG)。
对于一般的分子图（无向图），会先进行定向以创建DAG。
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import torch
import dgl
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger

logger = get_logger(__name__)

class TopoSerializer(BaseGraphSerializer):
    """拓扑排序序列化器"""
    
    def __init__(self):
        super().__init__()
        self.name = "topo"
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        初始化拓扑排序序列化器（统一接口实现）
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表（拓扑排序不需要统计信息）
        """
        # 保存数据集加载器引用
        self._dataset_loader = dataset_loader
        
        # 拓扑排序序列化器不需要预处理统计信息
        self._dataset_stats.update({
            'method': 'Topo',
            'description': '拓扑排序序列化',
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
        #将图转为有向图，边从小的id指向大的id
        src, dst = graph_data['dgl_graph'].edges()
        mask = src > dst
        src = src[mask]
        dst = dst[mask]
        # 保持与原图相同的节点数量，避免特征长度与节点数不一致
        orig = graph_data['dgl_graph']
        dgl_graph = dgl.graph((src, dst), num_nodes=int(orig.num_nodes()))
        # 兼容不同数据集的节点特征字段命名：优先 'feat'，其次 'attr'，再次 'feature'；不存在则不复制
        if 'feat' in orig.ndata:
            dgl_graph.ndata['feat'] = orig.ndata['feat']
        elif 'attr' in orig.ndata:
            dgl_graph.ndata['attr'] = orig.ndata['attr']
        elif 'feature' in orig.ndata:
            dgl_graph.ndata['feature'] = orig.ndata['feature']
        
        # 调用新的序列化逻辑，返回token和element序列
        token_sequence, element_sequence = self._topo_serialize(dgl_graph,graph_data['dgl_graph'])
        
        # 构建SerializationResult
        dgl_graph = graph_data['dgl_graph']
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _topo_serialize(self, dgl_graph: dgl.DGLGraph,raw_graph: dgl.DGLGraph) -> Tuple[List[int], List[str]]:
        """
        使用拓扑排序序列化图，返回token序列和element序列
        
        Args:
            mol_data: 图数据，包含dgl_graph
            dataset_loader: 数据集加载器，用于获取节点token表示
            start_node: 起始节点索引（拓扑排序通常不使用）
            
        Returns:
            Tuple[List[int], List[str]]: (token序列, element序列)
        """
        # 使用公共方法验证图数据
        num_nodes = dgl_graph.num_nodes()

        
        # 单节点特殊情况
        if num_nodes == 1:
            return [self.get_node_token(dgl_graph, 0)], ["node_0"]
        
        # 构建邻接列表
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        
        # 计算入度
        in_degree = [0] * num_nodes
        for neighbors in adj_list:
            for neighbor in neighbors:
                in_degree[neighbor] += 1
        
        # 拓扑排序
        result_token_ids = []
        result_element_ids = []
        queue = deque()
        
        # 将所有入度为0的节点加入队列
        for i in range(num_nodes):
            if in_degree[i] == 0:
                queue.append(i)
        
        # 执行拓扑排序
        while queue:
            # 按索引排序确保确定性
            queue = deque(sorted(queue))
            node = queue.popleft()
            node_tokens = self.get_node_token(raw_graph, node)
            result_token_ids.extend(node_tokens)
            if len(node_tokens) > 1:
                result_element_ids.append(f"START_NODE_{node}")
                for j, token in enumerate(node_tokens[1:-1]):
                      result_element_ids.append(f"node_{node}_dim_{j}")
                result_element_ids.append(f"END_NODE_{node}")
            else:
                result_element_ids.append(f"node_{node}")
            
            # 更新邻居的入度
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 如果还有节点未访问，说明存在环
        if len(result_token_ids) < num_nodes:
            logger.warning(f"拓扑排序失败，有环的图，添加剩余节点")
            # 对于有环的图，添加剩余节点
            for i in range(num_nodes):
                if i not in [j for j, _ in enumerate(result_token_ids)]:
                    node_tokens = self.get_node_token(raw_graph, i)
                    result_token_ids.extend(node_tokens)
                    if len(node_tokens) > 1:
                        result_element_ids.append(f"START_NODE_{i}")
                        for j, token in enumerate(node_tokens[1:-1]):
                              result_element_ids.append(f"node_{i}_dim_{j}")
                        result_element_ids.append(f"END_NODE_{i}")
                    else:
                        result_element_ids.append(f"node_{i}")
        
        return result_token_ids, result_element_ids
    