"""
中国邮递员问题序列化器 (基于NetworkX)
=====================================

基于中国邮递员问题(Chinese Postman Problem, CCP)的图序列化算法。
使用NetworkX的成熟算法来实现高效的CCP求解。
"""

from typing import Dict, Any, List, Tuple
import dgl
import networkx as nx
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger

# 设置logger
logger = get_logger(__name__)

class FCPPSerializer(BaseGraphSerializer):
    """中国邮递员问题序列化器 - 基于NetworkX的高效实现"""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True, verbose: bool = False):
        super().__init__()
        self.name = "freq_cpp"
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
        self.verbose = verbose
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """初始化序列化器"""
        self._dataset_loader = dataset_loader
        assert graph_data_list is not None, "❌ 图数据列表不能为空"
        logger.info(f"📊 FreqCPP开始收集全局统计信息，共 {len(graph_data_list)} 个图")
        self._collect_statistics_from_graphs(graph_data_list)
        
        # 预处理：标准化频率权重（基于全局统计）
        self._preprocess_frequency_weights()
        
        self._dataset_stats.update({
            'method': 'Frequency-Guided Chinese Postman Problem (NetworkX)',
            'description': '基于NetworkX的频率引导的中国邮递员问题序列化',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'algorithm': 'NetworkX based: Dijkstra + Max Weight Matching(edmonds-blossom algorithm) + Eulerian Circuit + Frequency-Guided'
        })
    
    def _preprocess_frequency_weights(self):
        """预处理：记录稀疏频率的基本统计（如果存在），用于日志与诊断。"""
        import torch as _torch
        if getattr(self, '_triplet_sparse_keys', None) is not None and getattr(self, '_triplet_sparse_vals', None) is not None:
            vals = self._triplet_sparse_vals.to(_torch.float32)
            if vals.numel() == 0:
                min_freq = 0.0
                max_freq = 0.0
                nnz = 0
            else:
                min_freq = float(vals.min().item())
                max_freq = float(vals.max().item())
                nnz = int(vals.numel())
            logger.info(f"📊 freq_cpp三元组频率(稀疏): 最小值={min_freq}, 最大值={max_freq}, 非零={nnz}")
            self._dataset_stats.update({
                'freq_min': min_freq,
                'freq_max': max_freq,
                'freq_normalized': False,
                'sparse_nnz': nnz
            })
        elif getattr(self, '_triplet_frequency_tensor', None) is not None:
            freq_tensor = self._triplet_frequency_tensor
            if freq_tensor.numel() == 0:
                min_freq = 0.0
                max_freq = 0.0
                nnz = 0
                mask_sum = 0
            else:
                mask = freq_tensor > 0
                vals = freq_tensor[mask].to(_torch.float32)
                min_freq = float(vals.min().item()) if vals.numel() > 0 else 0.0
                max_freq = float(vals.max().item()) if vals.numel() > 0 else 0.0
                mask_sum = int(mask.sum().item())
                nnz = mask_sum
            logger.info(f"📊 freq_cpp三元组频率(致密): 最小值={min_freq}, 最大值={max_freq}, 非零={nnz}")
            self._dataset_stats.update({
                'freq_min': min_freq,
                'freq_max': max_freq,
                'freq_normalized': True,
                'dense_nnz': nnz
            })
        else:
            logger.warning("⚠️ 未找到任何频率统计（稀疏/致密均为空）")
    
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
    
    def _convert_dgl_to_networkx(self, dgl_graph: dgl.DGLGraph) -> nx.MultiGraph:
        """将DGL图转换为NetworkX多重图，使用标准化的频率权重
        
        DGL中的双向边(u,v)和(v,u)代表一条无向边，但NetworkX会将它们视为重复。
        我们需要去重，只保留每条无向边的一个方向，并使用标准化的频率权重。
        """
        # 调用基类方法获取基础图结构
        G = super()._convert_dgl_to_networkx(dgl_graph)
        
        # 计算标准化的边权重（范围0~1），优先使用基类构建的归一化三元组频率张量
        edge_weights = self._calculate_edge_weights(dgl_graph)
        
        # 重新设置边权重
        for src, dst in G.edges():
            # 获取标准化频率权重
            normalized_freq = edge_weights[(src, dst)]
            
            # 权重设计：α * 长度权重 + (1-α) * 频率权重
            # α = 0.5 表示50%权重给回路长度，50%给频率引导
            alpha = 0.5  # 可调节的平衡参数
            
            # 长度权重：保持为1，确保回路长度最小化
            length_weight = 1.0
            
            # 频率权重：已经是0~1范围，高频率边权重接近1，需要反转
            # 反转：高频率边权重小（优先选择）
            freq_weight = 1.0 - normalized_freq
            
            # 组合权重
            edge_weight = alpha * length_weight + (1 - alpha) * freq_weight
            
            # 更新边权重
            G[src][dst][0]['weight'] = edge_weight
        
        return G
    
    def _calculate_edge_weights(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], float]:
        """计算图的边权重。优先使用基类的统一权重计算（已兼容稀疏/致密）。"""
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        base_weights = super()._calculate_edge_weights(dgl_graph)
        # 归一化到 0~1：按图内的最小/最大计数对 log10(count) 做线性缩放（仅用于 CPP 权重）
        if len(base_weights) == 0:
            return {}
        vals = list(base_weights.values())
        vmin = min(vals)
        vmax = max(vals)
        if vmax - vmin > 1e-12:
            norm = {k: (w - vmin) / (vmax - vmin) for k, w in base_weights.items()}
        else:
            norm = {k: 0.5 for k in base_weights.keys()}
        return norm
    
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
                total_weight = len(circuit)
                return total_weight, circuit
            except Exception:
                # 如果指定起始点失败，使用默认起始点
                circuit = list(nx.eulerian_circuit(graph))
                total_weight = len(circuit)  
                return total_weight, circuit
        
        # 3. 高效计算奇点对之间的最短路径（用 SciPy 稀疏最短路替代逐对 Dijkstra）
        pair_dist = {}
        pair_path = {}
        try:
            import numpy as _np
            import scipy.sparse as _sp
            from scipy.sparse.csgraph import shortest_path as _cs_shortest_path

            # 构建稀疏邻接矩阵（按 node index 顺序）
            nodes = list(graph.nodes())
            index_of = {n: i for i, n in enumerate(nodes)}
            rows = []
            cols = []
            weights = []
            for u, v, d in graph.edges(data=True):
                rows.append(index_of[u])
                cols.append(index_of[v])
                weights.append(float(d.get('weight', 1.0)))
                rows.append(index_of[v])
                cols.append(index_of[u])
                weights.append(float(d.get('weight', 1.0)))
            n = len(nodes)
            mat = _sp.csr_matrix((_np.array(weights), (_np.array(rows), _np.array(cols))), shape=(n, n))

            # 仅对奇点做源点，批量最短路
            sources = [index_of[u] for u in odd_nodes]
            dist_all, predecessors = _cs_shortest_path(mat, directed=False, indices=sources, return_predecessors=True)

            # 提取奇点对 (u,v) 的距离与路径
            for i, u in enumerate(odd_nodes):
                for j, v in enumerate(odd_nodes):
                    if i < j:
                        duv = float(dist_all[i, index_of[v]])
                        if _np.isinf(duv):
                            raise ValueError(f"❌ 无法找到节点{u}到{v}的路径")
                        pair_dist[(u, v)] = duv
                        # 回溯路径
                        path_rev = [index_of[v]]
                        cur = index_of[v]
                        while cur != index_of[u]:
                            cur = int(predecessors[i, cur])
                            if cur < 0:
                                raise ValueError(f"❌ 回溯路径失败: {u}->{v}")
                            path_rev.append(cur)
                        pair_path[(u, v)] = [nodes[k] for k in reversed(path_rev)]
        except Exception as e:
            raise ValueError(f"❌ SciPy 最短路计算失败: {e}")
        
        logger.debug(f"📊 高效计算了 {len(pair_dist)} 对奇度节点间的最短路径")
        logger.debug(f"🚀 Dijkstra调用次数: {len(odd_nodes)} (优化前需要 {len(pair_dist)} 次)")
        
        # 4. 构建完美匹配问题
        complete = nx.Graph()
        for (u, v), w in pair_dist.items():
            complete.add_edge(u, v, weight=w)  
        
        # 5. 求解最小权重完美匹配(O(K^3), K为奇度节点数)
        try:
            matched = nx.algorithms.matching.min_weight_matching(complete)# min必为max cardinality
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
                graph.add_edge(a, b, weight=1)# 边权为1的理由同上。不过这里之后也不会用到边的权重，所以无所谓
        
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
    