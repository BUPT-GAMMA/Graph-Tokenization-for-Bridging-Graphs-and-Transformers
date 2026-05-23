"""BFS (breadth-first search) serializer.
BFS（广度优先搜索）序列化器。"""

from typing import Dict, Any, List, Tuple, Optional
import dgl
from collections import deque
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger
import networkx as nx

logger = get_logger(__name__)

class BFSSerializer(BaseGraphSerializer):
    """BFS serializer."""
    
    def __init__(self):
        super().__init__()
        self.name = "bfs"
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize BFS serializer (no statistics needed)."""
        self._dataset_loader = dataset_loader
        self._dataset_stats.update({
            'method': 'BFS',
            'description': 'Breadth-first search serialization',
            'requires_statistics': False
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        token_sequence, element_sequence = self._BFS_serialize(graph_data, start_node=start_node)
        
        dgl_graph = graph_data['dgl_graph']
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _BFS_serialize(self, mol_data: Dict[str, Any], start_node: int = 0) -> Tuple[List[int], List[str]]:
        """BFS serialization."""
        dgl_graph  = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        # Single-node special case
        if num_nodes == 1:
            return [self.get_node_token(dgl_graph, 0)], ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "Start node index out of range"
        
        # Build adjacency list
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        
        # Execute BFS
        visited = [False] * num_nodes
        result_token_ids = []
        result_element_ids = []
        queue = deque()
        
        # Start BFS from start_node
        queue.append(start_node)
        visited[start_node] = True
        
        while queue:
            node = queue.popleft()
            node_tokens = self.get_node_token(dgl_graph, node)
            result_token_ids.extend(node_tokens)
            
            # Add element info for each token
            if len(node_tokens) > 1:
                result_element_ids.append(f"START_NODE_{node}")
                for j, token in enumerate(node_tokens[1:-1]):
                      result_element_ids.append(f"node_{node}_dim_{j}")
                result_element_ids.append(f"END_NODE_{node}")
            else:
                result_element_ids.append(f"node_{node}")
            
            # Sort neighbors by index for determinism
            neighbors = sorted(adj_list[node])
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        assert len(result_token_ids) == len(result_element_ids)
        # assert visited == [True] * num_nodes  # some graphs may be disconnected
        
        return result_token_ids, result_element_ids
    