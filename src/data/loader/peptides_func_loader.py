from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import gzip

import dgl
import numpy as np
import torch

from ..base_loader import BaseDataLoader
from config import ProjectConfig
from utils.logger import get_logger


logger = get_logger(__name__)


def reconstruct_dgl_graph_from_lightweight(graph_data: Dict[str, Any]) -> Tuple[dgl.DGLGraph, Any]:
    """Reconstruct a DGL graph from lightweight data."""
    # Reconstruct graph structure
    src, dst = graph_data['edges']
    g = dgl.graph((src, dst), num_nodes=graph_data['num_nodes'])
    
    # Restore features (preprocessing guarantees these fields)
    g.ndata['x'] = torch.from_numpy(graph_data['node_features'])
    g.edata['edge_attr'] = torch.from_numpy(graph_data['edge_features'])
    
    # Restore token IDs
    g.ndata['node_token_ids'] = torch.from_numpy(graph_data['node_token_ids'])
    g.edata['edge_token_ids'] = torch.from_numpy(graph_data['edge_token_ids'])
    
    # Restore type IDs
    g.ndata['node_type_id'] = torch.from_numpy(graph_data['node_type_ids'])
    g.edata['edge_type_id'] = torch.from_numpy(graph_data['edge_type_ids'])
    
    return g, None  # labels handled separately


class PeptidesFuncLoader(BaseDataLoader):
    """Peptides-func graph classification dataset (LRGB). Multi-label (10 functional labels)."""

    def __init__(self, config: ProjectConfig, dataset_name: str = "peptides_func", target_property: Optional[str] = None):
        super().__init__(dataset_name, config, target_property)
        self._all_data: Optional[List[Dict[str, Any]]] = None
        self._cache_built: bool = False
        self._node_attr_cache: Dict[int, Dict[int, int]] = {}
        self._edge_attr_cache: Dict[int, Dict[int, int]] = {}
        self._normalized_name = "peptides_func"
        self.load_data()

    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"Loading Peptides-func preprocessed data from: {self.data_dir}")
        train_index_file = self.data_dir / "train_index.json"
        test_index_file = self.data_dir / "test_index.json"
        val_index_file = self.data_dir / "val_index.json"
        for f in (train_index_file, test_index_file, val_index_file):
            if not f.exists():
                raise FileNotFoundError("Index files not found; run preprocessing first")
        # Read lightweight compressed format
        data_file = self.data_dir / "data.pkl.gz"
        if not data_file.exists():
            raise FileNotFoundError(f"Peptides-func data file not found: {data_file}")
        
        if self._all_data is None:
            with gzip.open(data_file, "rb") as f:
                raw_data: List[Tuple[Dict[str, Any], Any]] = pickle.load(f)
            
            all_data: List[Dict[str, Any]] = []
            for i, (graph_data, label_obj) in enumerate(raw_data):
                # Reconstruct DGL graph
                graph, _ = reconstruct_dgl_graph_from_lightweight(graph_data)
                
                # Labels: shape=(10,) numpy array from preprocessing
                assert isinstance(label_obj, np.ndarray) and label_obj.shape == (10,), f"Bad label format: {type(label_obj)}, shape={getattr(label_obj, 'shape', 'no-shape')}"
                labels = label_obj.tolist()
                
                sample = {
                    "id": f"{self._normalized_name}_{i}",
                    "dgl_graph": graph,
                    "num_nodes": int(graph.num_nodes()),
                    "num_edges": int(graph.num_edges()),
                    "properties": {"labels": labels},  # 10-dim binary vector
                    "dataset_name": self.dataset_name,
                    "data_type": "peptide_graph",
                }
                all_data.append(sample)
            
            self._all_data = all_data
            
        with open(train_index_file, "r") as f:
            train_indices = [int(x) for x in json.load(f)]
        with open(val_index_file, "r") as f:
            val_indices = [int(x) for x in json.load(f)]
        with open(test_index_file, "r") as f:
            test_indices = [int(x) for x in json.load(f)]
            
        train_data = [self._all_data[i] for i in train_indices if i < len(self._all_data)]
        val_data = [self._all_data[i] for i in val_indices if i < len(self._all_data)]
        test_data = [self._all_data[i] for i in test_indices if i < len(self._all_data)]
        
        return train_data, val_data, test_data

    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Extract 10-dim multi-labels."""
        return [s["properties"]["labels"] for s in data]

    def _get_data_metadata(self) -> Dict[str, Any]:
        if self._train_data is None:
            self.load_data()
        all_data = self._train_data + self._val_data + self._test_data
        if not all_data:
            return {}
        num_nodes = [s["num_nodes"] for s in all_data]
        num_edges = [s["num_edges"] for s in all_data]
        return {
            "dataset_name": self.dataset_name,
            "dataset_type": "peptide_graph",
            "total_graphs": len(all_data),
            "avg_num_nodes": float(np.mean(num_nodes)),
            "avg_num_edges": float(np.mean(num_edges)),
            "task_type": self.get_dataset_task_type(),
            "num_classes": self.get_num_classes(),
        }

    def get_dataset_task_type(self) -> str:
        return "multi_label_classification"

    def get_num_classes(self) -> int:
        return 10

    def get_default_target_property(self) -> Optional[str]:
        return "labels"

    def get_downstream_label_keys(self) -> List[str]:
        return ["labels"]

    def load_data(self):
        res = super().load_data()
        if self._all_data is not None and not self._cache_built:
            self._build_attribute_cache(self._all_data)
        return res

    def _build_attribute_cache(self, processed_data: List[Dict[str, Any]]):
        """Build attribute cache."""
        for sample in processed_data:
            g: dgl.DGLGraph = sample["dgl_graph"]
            gid = id(g)
            
            self._node_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.ndata["node_type_id"].tolist())}
            self._edge_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.edata["edge_type_id"].tolist())}
        self._cache_built = True

    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        if self._cache_built and id(graph) in self._node_attr_cache:
            return int(self._node_attr_cache[id(graph)][int(node_id)])
        return int(graph.ndata["node_type_id"][int(node_id)].item())

    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        if self._cache_built and id(graph) in self._edge_attr_cache:
            return int(self._edge_attr_cache[id(graph)][int(edge_id)])
        if "edge_type_id" in graph.edata:
            return int(graph.edata["edge_type_id"][int(edge_id)].item())
        return 0

    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        type_id = self.get_node_attribute(graph, node_id)
        return str(type_id)

    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        type_id = self.get_edge_attribute(graph, edge_id)
        return str(type_id)

    def get_most_frequent_edge_type(self) -> str:
        return "1"

    def get_edge_type_id_by_name(self, name: str) -> int:
        try:
            return int(name)
        except ValueError:
            return 0

    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        return [int(graph.ndata["node_token_ids"][int(node_id)][0].item())]

    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        return [int(graph.edata["edge_token_ids"][int(edge_id)][0].item())]

    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[List[int]]:
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        return graph.ndata["node_token_ids"][ids].tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[List[int]]:
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        return graph.edata["edge_token_ids"][ids].tolist()

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[str]:
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        return [str(int(v)) for v in graph.ndata["node_type_id"][ids].tolist()]

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[str]:
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        return [str(int(v)) for v in graph.edata["edge_type_id"][ids].tolist()]

    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.ndata["node_type_id"]

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.edata["edge_type_id"]

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.ndata["node_token_ids"].long()

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.edata["edge_token_ids"].long()

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        """Read token mappings from file."""
        token_mapping_file = self.data_dir / "token_mappings.json"
        
        with open(token_mapping_file, 'r') as f:
            mappings = json.load(f)
        
        token_map = {}
        
        # Node tokens (reverse-map from feature combo)
        for combo_str, token in mappings["node_combo_to_token"].items():
            type_id = (token - 1) // 2
            token_map[("node", type_id)] = token
        
        # Edge tokens (reverse-map from feature combo)
        for combo_str, token in mappings["edge_combo_to_token"].items():
            type_id = token // 2
            token_map[("edge", type_id)] = token
        
        return token_map
