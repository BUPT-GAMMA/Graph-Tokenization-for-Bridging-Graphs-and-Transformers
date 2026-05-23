"""
Topological sort serializer.
拓扑排序序列化器。

Serializes graphs via topological ordering. For undirected graphs,
edges are oriented (small ID -> large ID) to create a DAG first.
通过拓扑排序进行图序列化。对于无向图，先将边定向（小ID→大ID）构造DAG。
"""

from typing import Dict, Any, List, Optional, Tuple
from collections import deque

import torch
import dgl
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from utils.logger import get_logger

logger = get_logger(__name__)

class TopoSerializer(BaseGraphSerializer):
    """Topological sort serializer."""
    
    def __init__(self):
        super().__init__()
        self.name = "topo"
    
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize topological sort serializer (no statistics needed)."""
        self._dataset_loader = dataset_loader
        self._dataset_stats.update({
            'method': 'Topo',
            'description': 'Topological sort serialization',
            'requires_statistics': False
        })
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        # Orient edges: small ID -> large ID
        src, dst = graph_data['dgl_graph'].edges()
        mask = src > dst
        src = src[mask]
        dst = dst[mask]
        # Keep same node count as original to preserve feature alignment
        orig = graph_data['dgl_graph']
        dgl_graph = dgl.graph((src, dst), num_nodes=int(orig.num_nodes()))
        # Copy node features (try 'feat', 'attr', 'feature' in order)
        if 'feat' in orig.ndata:
            dgl_graph.ndata['feat'] = orig.ndata['feat']
        elif 'attr' in orig.ndata:
            dgl_graph.ndata['attr'] = orig.ndata['attr']
        elif 'feature' in orig.ndata:
            dgl_graph.ndata['feature'] = orig.ndata['feature']
        
        # Serialize and build result
        token_sequence, element_sequence = self._topo_serialize(dgl_graph,graph_data['dgl_graph'])
        
        dgl_graph = graph_data['dgl_graph']
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _topo_serialize(self, dgl_graph: dgl.DGLGraph,raw_graph: dgl.DGLGraph) -> Tuple[List[int], List[str]]:
        """Topological sort serialization."""
        num_nodes = dgl_graph.num_nodes()

        
        # Single-node special case
        if num_nodes == 1:
            return [self.get_node_token(dgl_graph, 0)], ["node_0"]
        
        # Build adjacency list
        adj_list = self._build_adjacency_list_from_dgl(dgl_graph)
        
        # Compute in-degrees
        in_degree = [0] * num_nodes
        for neighbors in adj_list:
            for neighbor in neighbors:
                in_degree[neighbor] += 1
        
        # Topological sort
        result_token_ids = []
        result_element_ids = []
        queue = deque()
        
        # Enqueue all zero in-degree nodes
        for i in range(num_nodes):
            if in_degree[i] == 0:
                queue.append(i)
        
        # Execute topological sort
        while queue:
            # Sort by index for determinism
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
            
            # Update neighbor in-degrees
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If nodes remain, graph has a cycle; append remaining nodes
        if len(result_token_ids) < num_nodes:
            logger.warning("Topological sort incomplete (cycle detected), appending remaining nodes")
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
    