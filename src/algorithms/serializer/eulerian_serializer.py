"""
欧拉回路序列化器
==============

基于欧拉回路的图序列化算法，确保遍历所有边且每条边只遍历一次。
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
import dgl
import torch
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger
import networkx as nx

logger = get_logger(__name__)

class EulerianSerializer(BaseGraphSerializer):
    """欧拉回路序列化器 - 严格实现欧拉回路算法"""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True):
        super().__init__()
        self.name = "eulerian"
        self.include_edge_tokens = include_edge_tokens  # 控制是否包含边token
        self.omit_most_frequent_edge = omit_most_frequent_edge  # 控制是否省略最高频边类型
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        初始化欧拉回路序列化器（统一接口实现）
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表（欧拉回路不需要统计信息）
        """
        # 保存数据集加载器引用
        self._dataset_loader = dataset_loader
        
        # 欧拉回路序列化器不需要预处理统计信息
        self._dataset_stats.update({
            'method': 'Eulerian',
            'description': '欧拉回路序列化',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge
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
            token_sequence, element_sequence = self._eulerian_serialize(graph_data, start_node=start_node)
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
                token_seq, element_seq = self._eulerian_serialize(subgraph_data, start_node=sub_start_node)
                
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
            assert isinstance(merged_token_sequence, list), f"merged_token_sequence: {merged_token_sequence}"
            assert all(isinstance(token, int) for token in merged_token_sequence), f"merged_token_sequence: {merged_token_sequence}"
            
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([merged_token_sequence], [merged_element_sequence], id_mapping)
    
    
    def _eulerian_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """
        使用欧拉回路序列化图，返回token序列和element序列
        
        Args:
            mol_data: 图数据，包含dgl_graph
            dataset_loader: 数据集加载器，用于获取节点token表示
            start_node: 起始节点索引
            
        Returns:
            Tuple[List[int], List[str]]: (token序列, element序列)
        """
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        if num_nodes == 0:
            raise ValueError("❌ 图没有节点，无法进行欧拉回路序列化")
        
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "❌ 起始节点索引超出范围"
        
        # 构建邻接列表并排序，避免依赖 DGL 内部边序
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        for i in range(len(adj_list)):
            adj_list[i].sort()
        
        # 检查是否存在欧拉回路
        if not self._has_eulerian_circuit(adj_list, num_nodes):
            adj_list = self._make_eulerian_by_doubling_edges(adj_list, num_nodes)
            # raise ValueError(f"❌ 图不是欧拉图，无法进行欧拉回路序列化。请使用CCP序列化器处理非欧拉图。")
        
        # 执行欧拉回路算法
        euler_path = self._find_eulerian_circuit(adj_list, start_node)
        
        if not euler_path:
            raise ValueError("❌ 无法找到欧拉回路")
        
        return self._convert_path_to_tokens(euler_path, mol_data)
  
    def _make_eulerian_by_doubling_edges(self, adj_list: List[List[int]], num_nodes: int) -> List[List[int]]:
        #添加反向边,创建新的邻接列表
        new_adj_list = [[] for _ in range(num_nodes)] 
        for i in range(num_nodes):
            for j in adj_list[i]:
                new_adj_list[i].append(j)
                new_adj_list[j].append(i)
        return new_adj_list

    
    def _has_eulerian_circuit(self, adj_list: List[List[int]], num_nodes: int) -> bool:
        """
        检查图是否存在欧拉回路
        
        对于无向图，节点的度数应该是它参与的边的总数。
        在我们的邻接列表表示中，每个节点的度数 = 出度 + 入度
        
        Args:
            adj_list: 邻接列表
            num_nodes: 节点数
            
        Returns:
            bool: 是否存在欧拉回路
        """
        # 计算每个节点的正确度数（出度 + 入度）
        degrees = [0] * num_nodes
        
        for i in range(num_nodes):
            # 出度：从节点i出发的边数
            degrees[i] += len(adj_list[i])
            
            # 入度：指向节点i的边数
            for neighbor in adj_list[i]:
                degrees[neighbor] += 1
        
        # 检查所有节点的度数是否为偶数
        for i in range(num_nodes):
            if degrees[i] % 2 != 0:
                # print(f"   节点{i}度数为{degrees[i]}（奇数）")
                return False
        
        # print(f"   所有节点度数均为偶数: {degrees}")
        
        # 检查图是否连通（使用无向图的连通性检查）
        visited = [False] * num_nodes
        
        # 找到第一个有边的节点作为起点
        start_node = 0
        for i in range(num_nodes):
            if degrees[i] > 0:
                start_node = i
                break
        
        # DFS检查连通性
        stack = [start_node]
        visited[start_node] = True
        
        while stack:
            node = stack.pop()
            # 检查邻居（双向）
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            
            # 还要检查指向当前节点的边
            for i in range(num_nodes):
                if node in adj_list[i] and not visited[i]:
                    visited[i] = True
                    stack.append(i)
        
        # 检查所有有边的节点是否都被访问到
        for i in range(num_nodes):
            if degrees[i] > 0 and not visited[i]:
                # print(f"   节点{i}不连通")
                return False
        
        # print(f"   图是连通的")
        return True
    
    def _find_eulerian_circuit(self, adj_list: List[List[int]], start_node: int) -> List[int]:
        """
        使用Hierholzer算法找到欧拉回路
        
        对于DGL图，每条边在邻接列表中都有明确表示：
        - 边(u,v)在adj_list[u]中存储v
        - 边(v,u)在adj_list[v]中存储u  
        使用一条边时，只移除这条边，不移除"反向边"（因为它是独立的边）
        
        Args:
            adj_list: 邻接列表
            num_nodes: 节点数
            start_node: 起始节点
            
        Returns:
            List[int]: 欧拉回路节点序列
        """
        # 复制邻接列表，因为我们需要修改它
        adj_copy = [neighbors[:] for neighbors in adj_list]
        
        # 使用Hierholzer算法
        circuit = []
        stack = [start_node]
        
        while stack:
            current = stack[-1]
            
            if adj_copy[current]:
                # 还有未使用的边
                next_node = adj_copy[current].pop()
                # 对于DGL图，不需要移除"反向边"，因为每条边都是独立存储的
                stack.append(next_node)
            else:
                # 没有未使用的边，回溯
                circuit.append(stack.pop())
        
        # 反转电路（因为我们是反向构建的）
        circuit.reverse()
        
        return circuit
