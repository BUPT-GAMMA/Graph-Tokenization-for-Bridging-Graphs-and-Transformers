"""
中国邮递员问题序列化器 (基于NetworkX)
=====================================

基于中国邮递员问题(Chinese Postman Problem, CCP)的图序列化算法。
使用NetworkX的成熟算法来实现高效的CCP求解。
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import dgl
from dgl.init import F
import torch
import networkx as nx
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger
import networkx as nx

logger = get_logger(__name__)

class CPPSerializer(BaseGraphSerializer):
    """中国邮递员问题序列化器 - 基于NetworkX的高效实现"""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True, verbose: bool = False):
        super().__init__()
        self.name = "cpp"
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
        self.verbose = verbose
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """初始化序列化器"""
        self._dataset_loader = dataset_loader
        
        self._dataset_stats.update({
            'method': 'Chinese Postman Problem (NetworkX)',
            'description': '基于NetworkX的中国邮递员问题序列化',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'algorithm': 'NetworkX based: Dijkstra + Max Weight Matching(edmonds-blossom algorithm) + eulerianian Circuit'
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """序列化单个图"""
        start_node = kwargs.get('start_node', 0)
        
        dgl_graph = self._validate_graph_data(graph_data)
        
        # 检查连通性
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # 连通图，使用原有逻辑
            token_sequence, element_sequence = self._CPP_serialize(graph_data, start_node=start_node)
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
                token_seq, element_seq = self._CPP_serialize(subgraph_data, start_node=sub_start_node)
                
                all_token_sequences.append(token_seq)
                all_element_sequences.append(element_seq)
            
            # 合并所有序列
            merged_token_sequence = []
            merged_element_sequence = []
            
            for i, (token_seq, element_seq) in enumerate(zip(all_token_sequences, all_element_sequences)):
                if i > 0:
                    # 在不同连通分量之间添加分隔符
                    logger.debug(f"🔗 连接第 {i} 个连通分量的序列")
                    
                    # 添加分隔符token
                    merged_token_sequence.append(self._dataset_loader.config.component_sep_token_id)
                    merged_element_sequence.append("component_sep")
                
                merged_token_sequence.extend(token_seq)
                merged_element_sequence.extend(element_seq)
            
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([merged_token_sequence], [merged_element_sequence], id_mapping)
    
    def _CPP_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """使用NetworkX实现的高效CCP算法"""
        
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        assert num_nodes > 0, "❌ 图没有节点"
        
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        # 1. 转换为NetworkX图
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        # 2. 检查连通性（现在由上层处理，这里确保连通）
        if not nx.is_connected(nx_graph):
            raise ValueError("❌ 内部错误：传入的图应该已经确保连通")
        
        # 3. 执行CCP算法
        try:
            total_weight, edge_path = self._chinese_postman_networkx(nx_graph, start_node)
            
            logger.debug(f"✅ CCP求解完成，路径长度: {len(edge_path)}")
            
        except Exception as e:
            raise ValueError(f"❌ CCP算法执行失败: {str(e)}")
        
        # 4. 转换为token序列
        # 构建节点路径
        node_path = [edge_path[0][0]]  # 起始节点
        for u, v in edge_path:
            node_path.append(v)
        
        return self._convert_path_to_tokens(node_path, mol_data)
    

    

    

    
    def _chinese_postman_networkx(self, graph: nx.MultiGraph, start_node: int = 0) -> Tuple[float, List[Tuple[int, int]]]:
        """
        基于NetworkX的中国邮递员问题求解
        
        Returns:
            (总权重, 边路径列表)
        """
        
        # 1. 找出所有奇度节点
        odd_nodes = [v for v in graph.nodes if graph.degree[v] % 2 == 1]
        
        logger.debug(f"🔍 找到 {len(odd_nodes)} 个奇度节点: {odd_nodes}")
        
        # 2. 如果已经是欧拉图，直接返回欧拉回路
        if len(odd_nodes) == 0:
            logger.debug("✅ 已经是欧拉图，直接构造欧拉回路")
            
            try:
                circuit = list(nx.eulerian_circuit(graph, source=start_node))
                total_weight = len(circuit)  # 边权为1，总权重=边数
                return total_weight, circuit
            except Exception as e:
                logger.debug(f"❌ 无法构造欧拉回路: {str(e)}")
                # 如果指定起始点失败，使用默认起始点
                circuit = list(nx.eulerian_circuit(graph))
                total_weight = len(circuit)  # 边权为1，总权重=边数
                return total_weight, circuit
        
        # 3. 高效计算奇点对之间的最短路径
        # 优化：对每个奇度节点计算一次到所有节点的最短路径，然后提取需要的部分
        # todo:using bellman-ford algorithm
        pair_dist = {}
        pair_path = {}
        
        # 为每个奇度节点计算到所有其他节点的最短路径
        odd_nodes_distances = {}
        odd_nodes_paths = {}
        
        for u in odd_nodes:
            try:
                # 一次性计算从u到所有节点的最短路径
                distances, paths = nx.single_source_dijkstra(graph, u, weight='weight')
                odd_nodes_distances[u] = distances
                odd_nodes_paths[u] = paths
            except Exception as e:
                # 如果Dijkstra失败，使用简单最短路径
                logger.debug(f"❌ 无法计算节点{u}的最短路径: {str(e)}")
                continue
        
        # 提取所有奇度节点对之间的距离和路径
        for i, u in enumerate(odd_nodes):
            for j, v in enumerate(odd_nodes):
                if i < j:  # 只计算无序对，避免重复
                    if v in odd_nodes_distances[u]:
                        pair_dist[(u, v)] = odd_nodes_distances[u][v]
                        pair_path[(u, v)] = odd_nodes_paths[u][v]
                    else:
                        raise ValueError(f"❌ 无法找到节点{u}到{v}的路径")
        
        logger.debug(f"📊 高效计算了 {len(pair_dist)} 对奇度节点间的最短路径")
        logger.debug(f"🚀 Dijkstra调用次数: {len(odd_nodes)} (优化前需要 {len(pair_dist)} 次)")
        
        # 4. 构建完美匹配问题
        complete = nx.Graph()
        for (u, v), w in pair_dist.items():
            complete.add_edge(u, v, weight=-w)  # 负权重用于最大权重匹配
        
        # 5. 求解最小权重完美匹配(O(K^3), K为奇度节点数)
        try:
            matched = nx.algorithms.matching.max_weight_matching(complete, maxcardinality=True)
            logger.debug(f"🔗 找到最优匹配: {matched}")
        except Exception as e:
            raise ValueError(f"❌ 完美匹配求解失败: {str(e)}")
        
        # 6. 在原图中添加匹配路径的边（创建增广图）
        for u, v in matched:
            # 获取最短路径
            path = pair_path.get((u, v)) or pair_path.get((v, u))
            if not path:
                continue
                
            # 在路径上添加额外的边（多重图会自动处理重复边）
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                # 对于MultiGraph，直接添加新边即可，NetworkX会自动处理多重边
                graph.add_edge(a, b, weight=1)
        
        # 7. 在增广图上构造欧拉回路
        assert nx.is_eulerian(graph), "❌ 增广图不是欧拉图，无法构造欧拉回路"
        assert start_node in graph.nodes, f"❌ 起始节点{start_node}不在增广图中"
        try:
            circuit = list(nx.eulerian_circuit(graph, source=start_node, keys=True))
            
            # 清理路径格式
            clean_circuit = []
            for edge_data in circuit:
                if len(edge_data) >= 2:
                    u, v = edge_data[0], edge_data[1]
                    clean_circuit.append((u, v))
            
            # 简单权重计算：每条边权重为1
            total_weight = len(clean_circuit)
            
            return total_weight, clean_circuit
            
        except Exception as e:
            raise ValueError(f"❌ 欧拉回路构造失败: {str(e)}")
    