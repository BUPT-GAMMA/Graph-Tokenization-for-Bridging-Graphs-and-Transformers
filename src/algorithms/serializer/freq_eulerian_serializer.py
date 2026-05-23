"""
Frequency-guided Eulerian circuit serializer.
频率引导的欧拉回路序列化器。

Uses triplet frequency as guidance signal for edge traversal order.
使用三元组频率作为边遍历顺序的引导信号。
Two-hop path stats are kept as commented-out placeholders (disabled by design).
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

logger = get_logger(__name__)


class FeulerSerializer(BaseGraphSerializer):
    """Frequency-guided Eulerian circuit serializer."""
    
    def __init__(self, verbose: bool = False, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True):
        super().__init__()
        self.name = "feuler"
        self.triplet_frequencies = defaultdict(int)
        self.two_hop_frequencies = defaultdict(int)
        self.statistics_collected = False
        self.verbose = verbose
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
        
        # Runtime stats
        self.serialization_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'avg_tokens_per_molecule': 0.0
        }
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize frequency-guided Eulerian serializer."""
        assert graph_data_list is not None, "Graph data list must not be empty"
        logger.info(f"FEuler collecting global stats from {len(graph_data_list)} graphs")
        self._collect_statistics_from_graphs(graph_data_list)
        
        self._dataset_stats.update({
            'statistics_collected': self.statistics_collected,
            'triplet_frequencies_count': len(self.triplet_frequencies),
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge,
            'two_hop_frequencies_enabled': False,
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        dgl_graph = self._validate_graph_data(graph_data)
        
        # Check connectivity
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # Connected: standard logic
            token_sequence, element_sequence = self._gseq_serialize(graph_data, start_node)
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
                token_seq, element_seq = self._gseq_serialize(subgraph_data, sub_start_node)
                
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
    
    def _gseq_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """Frequency-guided Eulerian circuit serialization."""
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        # Single-node special case
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "Start node index out of range"
        
        # Find frequency-guided Eulerian circuit
        eulerian_path = self._find_frequency_guided_eulerian_circuit(dgl_graph, start_node)
        
        if not eulerian_path:
            raise ValueError("Cannot find frequency-guided Eulerian circuit")
        
        return self._convert_path_to_tokens(eulerian_path, mol_data)
    
    def _find_frequency_guided_eulerian_circuit(self, dgl_graph: dgl.DGLGraph, start_node: int = 0) -> Optional[List[int]]:
        """Find frequency-guided Eulerian circuit directly from DGL graph."""
        # Compute edge frequency weights
        edge_weights = self._calculate_edge_weights(dgl_graph)
        
        # Build weighted adjacency list
        weighted_graph = self._build_weighted_adjacency_list(dgl_graph, edge_weights)
        
        # Ensure start node has outgoing edges
        if start_node not in weighted_graph or not weighted_graph[start_node]:
            for i in range(dgl_graph.num_nodes()):
                if weighted_graph[i]:
                    start_node = i
                    break
            else:
                raise ValueError("No node has outgoing edges")
        
        logger.debug(f"Finding frequency-guided Eulerian circuit from node {start_node}")
        
        # Frequency-guided Hierholzer's algorithm
        return self._frequency_guided_hierholzer(weighted_graph, start_node)
    
    def _build_weighted_adjacency_list(self, dgl_graph: dgl.DGLGraph, 
                                     edge_weights: Dict[Tuple[int, int], float]) -> Dict[int, List[Tuple[int, float]]]:
        """Build weighted adjacency list from DGL graph and edge weights."""
        weighted_graph = defaultdict(list)
        
        # Get all edges
        src_nodes, dst_nodes = self._get_all_edges_from_heterograph(dgl_graph)
        
        # Build adjacency list
        for src, dst in zip(src_nodes.numpy(), dst_nodes.numpy()):
            src, dst = int(src), int(dst)
            weight = edge_weights[(src, dst)]
            weighted_graph[src].append((dst, weight))
        
        # Sort by weight desc, neighbor ID asc (stable tie-breaker)
        for node in weighted_graph:
            weighted_graph[node].sort(key=lambda x: (-x[1], x[0]))
        
        return weighted_graph
    
    def _frequency_guided_hierholzer(self, weighted_graph: Dict[int, List[Tuple[int, float]]], 
                                   start_node: int) -> List[int]:
        """Frequency-guided Hierholzer's algorithm. Prefers high-frequency edges."""
        # Copy graph (avoid modifying original)
        graph = defaultdict(list)
        for node, neighbors in weighted_graph.items():
            graph[node] = neighbors.copy()
        
        # Hierholzer main loop
        current_path = [start_node]
        circuit = []
        current_node = start_node
        
        while current_path:
            if graph[current_node]:
                # Pick highest-weight edge (list pre-sorted by (-weight, neighbor_id))
                next_node, weight = graph[current_node].pop(0)
                current_path.append(next_node)
                current_node = next_node
            else:
                # No unused edges: add to circuit
                circuit.append(current_path.pop())
                if current_path:
                    current_node = current_path[-1]
        
        # Reverse to get correct order
        circuit.reverse()
        
        logger.debug(f"Found frequency-guided Eulerian circuit: length={len(circuit)}")
        return circuit

    