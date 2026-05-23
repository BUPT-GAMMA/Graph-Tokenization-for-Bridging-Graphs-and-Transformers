"""DFS (depth-first search) serializer.
DFS（深度优先搜索）序列化器。"""

from ast import Tuple
from typing import Dict, Any, List
import dgl
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger
import networkx as nx

logger = get_logger(__name__)


class DFSSerializer(BaseGraphSerializer):
    """DFS serializer."""
    
    def __init__(self):
        super().__init__()
        self.name = "dfs"
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize DFS serializer (no statistics needed)."""
        self._dataset_loader = dataset_loader
        self._dataset_stats.update({
            'method': 'DFS',
            'description': 'Depth-first search serialization',
            'requires_statistics': False
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        start_node = kwargs.get('start_node', 0)
        token_sequence, element_sequence = self._DFS_serialize(graph_data, start_node=start_node)
        
        dgl_graph = graph_data['dgl_graph']
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _DFS_serialize(self, mol_data, start_node=0):
        """DFS serialization."""
        # TODO: not yet refactored; similar to BFS but with DFS traversal
        dgl_graph = self._validate_graph_data(mol_data)
        num_nodes = dgl_graph.num_nodes()
        
        # Single-node special case
        if num_nodes == 1:
            return [self.get_node_token(dgl_graph, 0)], ["node_0"]
        
        assert start_node >= 0 and start_node < num_nodes, "Start node index out of range"

        # Build adjacency list
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        
        # Execute DFS
        visited = [False] * num_nodes
        result_token_ids = []
        result_element_ids = []
        
        def dfs(node):
            visited[node] = True
            node_tokens = self.get_node_token(dgl_graph, node)
            result_token_ids.extend(node_tokens)
            if len(node_tokens) > 1:
                result_element_ids.append(f"START_NODE_{node}")
                for j, token in enumerate(node_tokens[1:-1]):
                      result_element_ids.append(f"node_{node}_dim_{j}")
                result_element_ids.append(f"END_NODE_{node}")
            else:
                result_element_ids.append(f"node_{node}")
            
            for neighbor in adj_list[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Start DFS from start_node
        dfs(start_node)
        
        # Handle disconnected components
        for i in range(num_nodes):
            if not visited[i]:
                dfs(i)
        
        return result_token_ids, result_element_ids
    
 