"""
Chinese Postman Problem serializer (NetworkX-based).
中国邮递员问题序列化器（基于NetworkX）。
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
    """Chinese Postman Problem serializer using NetworkX."""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True, verbose: bool = False):
        super().__init__()
        self.name = "cpp"
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
        self.verbose = verbose
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize serializer."""
        self._dataset_loader = dataset_loader
        
        self._dataset_stats.update({
            'method': 'Chinese Postman Problem (NetworkX)',
            'description': 'Chinese Postman Problem serialization (NetworkX)',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'algorithm': 'NetworkX based: Dijkstra + Max Weight Matching(edmonds-blossom algorithm) + eulerianian Circuit'
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        
        dgl_graph = self._validate_graph_data(graph_data)
        
        # Check connectivity
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # Connected: standard logic
            token_sequence, element_sequence = self._CPP_serialize(graph_data, start_node=start_node)
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([token_sequence], [element_sequence], id_mapping)
        else:
            self._current_edge_id_mapping=None
            # Disconnected: process each component
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
        
        # 2. Ensure connectivity
        if not nx.is_connected(nx_graph):
            raise ValueError("Internal error: graph should already be connected")
        
        # 3. Run CPP algorithm
        try:
            total_weight, edge_path = self._chinese_postman_networkx(nx_graph, start_node)
            logger.debug(f"CPP solved, path length: {len(edge_path)}")
        except Exception as e:
            raise ValueError(f"CPP algorithm failed: {str(e)}")
        
        # 4. Convert to token sequence
        node_path = [edge_path[0][0]]
        for u, v in edge_path:
            node_path.append(v)
        
        return self._convert_path_to_tokens(node_path, mol_data)
    

    

    

    
    def _chinese_postman_networkx(self, graph: nx.MultiGraph, start_node: int = 0) -> Tuple[float, List[Tuple[int, int]]]:
        """Solve Chinese Postman Problem using NetworkX."""
        
        # 1. Find all odd-degree nodes
        odd_nodes = [v for v in graph.nodes if graph.degree[v] % 2 == 1]
        
        logger.debug(f"Found {len(odd_nodes)} odd-degree nodes: {odd_nodes}")
        
        # 2. Already Eulerian: build circuit directly
        if len(odd_nodes) == 0:
            logger.debug("Already Eulerian, building circuit directly")
            
            try:
                circuit = list(nx.eulerian_circuit(graph, source=start_node))
                total_weight = len(circuit)
                return total_weight, circuit
            except Exception as e:
                logger.debug(f"Eulerian circuit failed with start node: {str(e)}")
                # Fallback to default start node
                circuit = list(nx.eulerian_circuit(graph))
                total_weight = len(circuit)
                return total_weight, circuit
        
        # 3. Compute shortest paths between odd-node pairs
        # todo:using bellman-ford algorithm
        pair_dist = {}
        pair_path = {}
        
        # Single-source Dijkstra from each odd node
        odd_nodes_distances = {}
        odd_nodes_paths = {}
        
        for u in odd_nodes:
            try:
                # Compute shortest paths from u to all nodes
                distances, paths = nx.single_source_dijkstra(graph, u, weight='weight')
                odd_nodes_distances[u] = distances
                odd_nodes_paths[u] = paths
            except Exception as e:
                logger.debug(f"Dijkstra failed for node {u}: {str(e)}")
                continue
        
        # Extract pairwise distances and paths
        for i, u in enumerate(odd_nodes):
            for j, v in enumerate(odd_nodes):
                if i < j:
                    if v in odd_nodes_distances[u]:
                        pair_dist[(u, v)] = odd_nodes_distances[u][v]
                        pair_path[(u, v)] = odd_nodes_paths[u][v]
                    else:
                        raise ValueError(f"No path found between node {u} and {v}")
        
        logger.debug(f"Computed {len(pair_dist)} odd-node pair shortest paths")
        logger.debug(f"Dijkstra calls: {len(odd_nodes)} (was {len(pair_dist)} before optimization)")
        
        # 4. Build perfect matching problem
        complete = nx.Graph()
        for (u, v), w in pair_dist.items():
            complete.add_edge(u, v, weight=-w)  # negate for max-weight matching
        
        # 5. Min-weight perfect matching (O(K^3))
        try:
            matched = nx.algorithms.matching.max_weight_matching(complete, maxcardinality=True)
            logger.debug(f"Found optimal matching: {matched}")
        except Exception as e:
            raise ValueError(f"Perfect matching failed: {str(e)}")
        
        # 6. Augment graph with matched paths
        for u, v in matched:
            path = pair_path.get((u, v)) or pair_path.get((v, u))
            if not path:
                continue
                
            # Add extra edges along the path
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
    