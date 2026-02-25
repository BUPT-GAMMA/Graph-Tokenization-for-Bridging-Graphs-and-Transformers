"""
Eulerian circuit serializer.
欧拉回路序列化器。

Graph serialization via Eulerian circuit: traverses every edge exactly once.
通过欧拉回路进行图序列化：恰好遍历每条边一次。
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
    """Eulerian circuit serializer."""
    
    def __init__(self, include_edge_tokens: bool = True, omit_most_frequent_edge: bool = True):
        super().__init__()
        self.name = "eulerian"
        self.include_edge_tokens = include_edge_tokens
        self.omit_most_frequent_edge = omit_most_frequent_edge
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize Eulerian serializer (no statistics needed)."""
        self._dataset_loader = dataset_loader
        self._dataset_stats.update({
            'method': 'Eulerian',
            'description': 'Eulerian circuit serialization',
            'requires_statistics': False,
            'include_edge_tokens': self.include_edge_tokens,
            'omit_most_frequent_edge': self.omit_most_frequent_edge
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        
        dgl_graph = self._validate_graph_data(graph_data)
        
        # Check connectivity
        nx_graph = self._convert_dgl_to_networkx(dgl_graph)
        
        if nx.is_connected(nx_graph):
            # Connected: standard logic
            token_sequence, element_sequence = self._eulerian_serialize(graph_data, start_node=start_node)
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
                token_seq, element_seq = self._eulerian_serialize(subgraph_data, start_node=sub_start_node)
                
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
            assert isinstance(merged_token_sequence, list), f"merged_token_sequence: {merged_token_sequence}"
            assert all(isinstance(token, int) for token in merged_token_sequence), f"merged_token_sequence: {merged_token_sequence}"
            
            id_mapping = GlobalIDMapping(dgl_graph)
            return SerializationResult([merged_token_sequence], [merged_element_sequence], id_mapping)
    
    
    def _eulerian_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """Serialize graph via Eulerian circuit."""
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        if num_nodes == 0:
            raise ValueError("Graph has no nodes")
        
        if num_nodes == 1:
            return self.get_node_token(dgl_graph, 0), ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "Start node index out of range"
        
        # Build adjacency list and sort for determinism
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        for i in range(len(adj_list)):
            adj_list[i].sort()
        
        # Check for Eulerian circuit existence
        if not self._has_eulerian_circuit(adj_list, num_nodes):
            adj_list = self._make_eulerian_by_doubling_edges(adj_list, num_nodes)
            # raise ValueError("Graph is not Eulerian. Use CPP serializer for non-Eulerian graphs.")
        
        # Run Eulerian circuit algorithm
        euler_path = self._find_eulerian_circuit(adj_list, start_node)
        
        if not euler_path:
            raise ValueError("Cannot find Eulerian circuit")
        
        return self._convert_path_to_tokens(euler_path, mol_data)
  
    def _make_eulerian_by_doubling_edges(self, adj_list: List[List[int]], num_nodes: int) -> List[List[int]]:
        # Add reverse edges to make graph Eulerian
        new_adj_list = [[] for _ in range(num_nodes)] 
        for i in range(num_nodes):
            for j in adj_list[i]:
                new_adj_list[i].append(j)
                new_adj_list[j].append(i)
        return new_adj_list

    
    def _has_eulerian_circuit(self, adj_list: List[List[int]], num_nodes: int) -> bool:
        """Check if graph has an Eulerian circuit (all degrees even, graph connected)."""
        # Compute degree (out-degree + in-degree)
        degrees = [0] * num_nodes
        
        for i in range(num_nodes):
            # Out-degree
            degrees[i] += len(adj_list[i])
            
            # In-degree
            for neighbor in adj_list[i]:
                degrees[neighbor] += 1
        
        # Check all degrees are even
        for i in range(num_nodes):
            if degrees[i] % 2 != 0:
                # print(f"   Node {i} degree={degrees[i]} (odd)")
                return False
        
        # print(f"   All degrees even: {degrees}")
        
        # Check connectivity
        visited = [False] * num_nodes
        
        # Find first node with edges as start
        start_node = 0
        for i in range(num_nodes):
            if degrees[i] > 0:
                start_node = i
                break
        
        # DFS connectivity check
        stack = [start_node]
        visited[start_node] = True
        
        while stack:
            node = stack.pop()
            # Check neighbors (both directions)
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
            
            # Also check edges pointing to current node
            for i in range(num_nodes):
                if node in adj_list[i] and not visited[i]:
                    visited[i] = True
                    stack.append(i)
        
        # Verify all nodes with edges were visited
        for i in range(num_nodes):
            if degrees[i] > 0 and not visited[i]:
                # print(f"   Node {i} not connected")
                return False
        
        # print(f"   Graph is connected")
        return True
    
    def _find_eulerian_circuit(self, adj_list: List[List[int]], start_node: int) -> List[int]:
        """
        Find Eulerian circuit using Hierholzer's algorithm.
        
        For DGL graphs, each edge is explicitly represented in the adjacency list:
        - Edge (u,v) is stored as v in adj_list[u]
        - Edge (v,u) is stored as u in adj_list[v]
        When using an edge, only remove the edge, not the "reverse edge" (since it's an independent edge)
        
        Args:
            adj_list: Adjacency list
            start_node: Starting node
            
        Returns:
            List[int]: Eulerian circuit node sequence
        """
        # Copy adjacency list (will be modified)
        adj_copy = [neighbors[:] for neighbors in adj_list]
        
        # Hierholzer's algorithm
        circuit = []
        stack = [start_node]
        
        while stack:
            current = stack[-1]
            
            if adj_copy[current]:
                # Unused edges remain
                next_node = adj_copy[current].pop()
                # No need to remove reverse edge; each DGL edge is stored independently
                stack.append(next_node)
            else:
                # No unused edges; backtrack
                circuit.append(stack.pop())
        
        # Reverse (built in reverse order)
        circuit.reverse()
        
        return circuit
