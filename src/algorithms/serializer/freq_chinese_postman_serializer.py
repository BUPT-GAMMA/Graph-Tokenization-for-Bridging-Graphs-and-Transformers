"""
Frequency-guided Chinese Postman Problem serializer (NetworkX-based).
频率引导的中国邮递员问题序列化器（基于NetworkX）。
"""

from typing import Dict, Any, List, Tuple
import dgl
import networkx as nx
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger

logger = get_logger(__name__)

class FCPPSerializer(BaseGraphSerializer):
    """Frequency-guided Chinese Postman serializer using NetworkX."""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True, verbose: bool = False):
        super().__init__()
        self.name = "freq_cpp"
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
        self.verbose = verbose
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize serializer."""
        self._dataset_loader = dataset_loader
        assert graph_data_list is not None, "Graph data list must not be empty"
        logger.info(f"FreqCPP collecting global stats from {len(graph_data_list)} graphs")
        self._collect_statistics_from_graphs(graph_data_list)
        
        # Preprocess: normalize frequency weights from global stats
        self._preprocess_frequency_weights()
        
        self._dataset_stats.update({
            'method': 'Frequency-Guided Chinese Postman Problem (NetworkX)',
            'description': 'Frequency-guided Chinese Postman Problem serialization (NetworkX)',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'algorithm': 'NetworkX based: Dijkstra + Max Weight Matching(edmonds-blossom algorithm) + Eulerian Circuit + Frequency-Guided'
        })
    
    def _preprocess_frequency_weights(self):
        """Log basic stats of sparse/dense triplet frequencies for diagnostics."""
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
            logger.info(f"freq_cpp triplet freq (sparse): min={min_freq}, max={max_freq}, nnz={nnz}")
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
            logger.info(f"freq_cpp triplet freq (dense): min={min_freq}, max={max_freq}, nnz={nnz}")
            self._dataset_stats.update({
                'freq_min': min_freq,
                'freq_max': max_freq,
                'freq_normalized': True,
                'dense_nnz': nnz
            })
        else:
            logger.warning("No frequency stats found (sparse/dense both empty)")
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        
        dgl_graph = self._validate_graph_data(graph_data)
        
        # Check connectivity
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # Connected graph: use standard logic
            token_sequence, element_sequence = self._CPP_serialize(graph_data, start_node=start_node)
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([token_sequence], [element_sequence], id_mapping)
        else:
            self._current_edge_id_mapping=None
            # Disconnected: process each component separately
            logger.debug("Graph disconnected, processing components separately")
            
            subgraphs = self._split_connected_components(dgl_graph)
            logger.debug(f"Found {len(subgraphs)} connected components")
            
            all_token_sequences = []
            all_element_sequences = []
            
            for i, subgraph in enumerate(subgraphs):
                logger.debug(f"Processing component {i+1}/{len(subgraphs)}, nodes: {subgraph.num_nodes()}")
                
                subgraph_data = {
                    'dgl_graph': subgraph,
                    **{k: v for k, v in graph_data.items() if k != 'dgl_graph'}
                }
                
                sub_start_node = 0
                token_seq, element_seq = self._CPP_serialize(subgraph_data, start_node=sub_start_node)
                
                all_token_sequences.append(token_seq)
                all_element_sequences.append(element_seq)
            
            # Merge all sequences
            merged_token_sequence = []
            merged_element_sequence = []
            
            for i, (token_seq, element_seq) in enumerate(zip(all_token_sequences, all_element_sequences)):
                if i > 0:
                    logger.debug(f"Joining component {i} sequence")
                    
                    merged_token_sequence.append(self._dataset_loader.config.component_sep_token_id)
                    merged_element_sequence.append("component_sep")
                
                merged_token_sequence.extend(token_seq)
                merged_element_sequence.extend(element_seq)
            
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([merged_token_sequence], [merged_element_sequence], id_mapping)
    
    def _CPP_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """CPP serialization using NetworkX."""
        
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        assert num_nodes > 0, "Graph has no nodes"
        
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        # 1. Convert to NetworkX graph
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        # 2. Ensure connectivity (caller should have split components)
        if not nx.is_connected(nx_graph):
            raise ValueError("Internal error: graph should already be connected")
        
        # 3. Run CPP algorithm
        try:
            total_weight, edge_path = self._chinese_postman_networkx(nx_graph, start_node)
            
            logger.debug(f"CPP solved, path length: {len(edge_path)}")
            
        except Exception as e:
            raise ValueError(f"CPP algorithm failed: {str(e)}")
        
        # 4. Convert to token sequence
        node_path = [edge_path[0][0]]  # start node
        for u, v in edge_path:
            node_path.append(v)
        
        return self._convert_path_to_tokens(node_path, mol_data)
    
    def _convert_dgl_to_networkx(self, dgl_graph: dgl.DGLGraph) -> nx.MultiGraph:
        """Convert DGL graph to NetworkX MultiGraph with normalized frequency weights."""
        # Get base graph structure from parent
        G = super()._convert_dgl_to_networkx(dgl_graph)
        
        # Compute normalized edge weights (0~1)
        edge_weights = self._calculate_edge_weights(dgl_graph)
        
        # Update edge weights
        for src, dst in G.edges():
            normalized_freq = edge_weights[(src, dst)]
            
            # Weight = alpha * length + (1-alpha) * frequency
            alpha = 0.5
            length_weight = 1.0
            # Invert: high freq -> low weight (prefer frequent edges)
            freq_weight = 1.0 - normalized_freq
            edge_weight = alpha * length_weight + (1 - alpha) * freq_weight
            G[src][dst][0]['weight'] = edge_weight
        
        return G
    
    def _calculate_edge_weights(self, dgl_graph: dgl.DGLGraph) -> Dict[Tuple[int, int], float]:
        """Compute edge weights, normalized to [0,1]."""
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        base_weights = super()._calculate_edge_weights(dgl_graph)
        # Normalize to [0,1] by linear scaling of log10(count) within graph
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
        Solve Chinese Postman Problem using NetworkX.
        
        Returns:
            (total_weight, edge_path_list)
        """
        
        # 1. Find all odd-degree nodes
        odd_nodes = [v for v in graph.nodes if graph.degree[v] % 2 == 1]
        
        logger.debug(f"Found {len(odd_nodes)} odd-degree nodes: {odd_nodes}")
        
        # 2. If already Eulerian, directly build Eulerian circuit
        if len(odd_nodes) == 0:
            logger.debug("Already Eulerian, building circuit directly")
            
            try:
                circuit = list(nx.eulerian_circuit(graph, source=start_node))
                total_weight = len(circuit)
                return total_weight, circuit
            except Exception:
                # Fallback to default start node
                circuit = list(nx.eulerian_circuit(graph))
                total_weight = len(circuit)  
                return total_weight, circuit
        
        # 3. Compute shortest paths between odd-node pairs (SciPy sparse)
        pair_dist = {}
        pair_path = {}
        try:
            import numpy as _np
            import scipy.sparse as _sp
            from scipy.sparse.csgraph import shortest_path as _cs_shortest_path

            # Build sparse adjacency matrix
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

            # Batch shortest paths from odd nodes only
            sources = [index_of[u] for u in odd_nodes]
            dist_all, predecessors = _cs_shortest_path(mat, directed=False, indices=sources, return_predecessors=True)

            # Extract odd-node pair distances and paths
            for i, u in enumerate(odd_nodes):
                for j, v in enumerate(odd_nodes):
                    if i < j:
                        duv = float(dist_all[i, index_of[v]])
                        if _np.isinf(duv):
                            raise ValueError(f"No path found between node {u} and {v}")
                        pair_dist[(u, v)] = duv
                        # Backtrace path
                        path_rev = [index_of[v]]
                        cur = index_of[v]
                        while cur != index_of[u]:
                            cur = int(predecessors[i, cur])
                            if cur < 0:
                                raise ValueError(f"Path backtrace failed: {u}->{v}")
                            path_rev.append(cur)
                        pair_path[(u, v)] = [nodes[k] for k in reversed(path_rev)]
        except Exception as e:
            raise ValueError(f"SciPy shortest path computation failed: {e}")
        
        logger.debug(f"Computed {len(pair_dist)} odd-node pair shortest paths")
        logger.debug(f"Dijkstra calls: {len(odd_nodes)} (was {len(pair_dist)} before optimization)")
        
        # 4. Build perfect matching problem
        complete = nx.Graph()
        for (u, v), w in pair_dist.items():
            complete.add_edge(u, v, weight=w)  
        
        # 5. Min-weight perfect matching (O(K^3), K = odd-degree node count)
        try:
            matched = nx.algorithms.matching.min_weight_matching(complete)
            logger.debug(f"Found optimal matching: {matched}")
        except Exception as e:
            raise ValueError(f"Perfect matching failed: {str(e)}")
        
        # 6. Augment original graph with matched paths
        for u, v in matched:
            path = pair_path.get((u, v)) or pair_path.get((v, u))
            if not path:
                continue
                
            path = pair_path.get((u, v)) or pair_path.get((v, u))
            if not path:
                continue
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                graph.add_edge(a, b, weight=1)
        
        # 7. Build Eulerian circuit on augmented graph
        assert nx.is_eulerian(graph), "Augmented graph is not Eulerian"
        assert start_node in graph.nodes, f"Start node {start_node} not in augmented graph"
        try:
            circuit = list(nx.eulerian_circuit(graph, source=start_node, keys=True))
            
            # Clean path format
            clean_circuit = []
            for edge_data in circuit:
                if len(edge_data) >= 2:
                    u, v = edge_data[0], edge_data[1]
                    clean_circuit.append((u, v))
            
            # Simple weight: 1 per edge
            total_weight = len(clean_circuit)
            
            return total_weight, clean_circuit
            
        except Exception as e:
            raise ValueError(f"Eulerian circuit construction failed: {str(e)}")
    