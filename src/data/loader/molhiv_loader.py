from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple
import json

import dgl
import numpy as np
import torch

from ..base_loader import BaseDataLoader
from config import ProjectConfig
from utils.logger import get_logger


logger = get_logger(__name__)


class MOLHIVLoader(BaseDataLoader):
    """ogbg-molhiv graph classification dataset (OGB).

    Node tokens: atomic number mapped to odd domain (2Z+1).
    Edge tokens: bond type ID mapped to even domain (2E).
    """

    # Aligned with ZINC/QM9
    BOND_TYPES = {0: 'NONE', 1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE', 4: 'AROMATIC'}

    def __init__(self, config: ProjectConfig, dataset_name: str = "molhiv", target_property: Optional[str] = None):
        super().__init__(dataset_name, config, target_property)
        self._all_data: Optional[List[Dict[str, Any]]] = None
        self._cache_built: bool = False
        self._node_attr_cache: Dict[int, Dict[int, int]] = {}
        self._edge_attr_cache: Dict[int, Dict[int, int]] = {}
        self._normalized_name = "molhiv"
        self.load_data()

    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"Loading MOLHIV preprocessed data from: {self.data_dir}")
        train_index_file = self.data_dir / "train_index.json"
        test_index_file = self.data_dir / "test_index.json"
        val_index_file = self.data_dir / "val_index.json"
        for f in (train_index_file, test_index_file, val_index_file):
            if not f.exists():
                raise FileNotFoundError("Index files not found; run preprocessing first")
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        if self._all_data is None:
            with open(data_file, "rb") as f:
                raw_data: List[Tuple[dgl.DGLGraph, Any]] = pickle.load(f)
            all_data: List[Dict[str, Any]] = []
            for i, (graph, label_obj) in enumerate(raw_data):
                sample = {
                    "id": f"{self._normalized_name}_{i}",
                    "dgl_graph": graph,
                    "num_nodes": int(graph.num_nodes()),
                    "num_edges": int(graph.num_edges()),
                    "properties": {"label": int(label_obj) if isinstance(label_obj, (int, np.integer)) else int(label_obj)},
                    "dataset_name": self.dataset_name,
                    "data_type": "molecule_graph",
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
        return [int(s.get("properties", {}).get("label", 0)) for s in data]

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
            "dataset_type": "molecule_graph",
            "total_graphs": len(all_data),
            "avg_num_nodes": float(np.mean(num_nodes)),
            "avg_num_edges": float(np.mean(num_edges)),
            "task_type": self.get_dataset_task_type(),
            "num_classes": self.get_num_classes(),
        }

    def load_data(self):
        res = super().load_data()
        if self._all_data is not None and not self._cache_built:
            self._build_attribute_cache(self._all_data)
        return res

    def _build_attribute_cache(self, processed_data: List[Dict[str, Any]]):
        for sample in processed_data:
            g: dgl.DGLGraph = sample["dgl_graph"]
            gid = id(g)
            # Preprocessing provides node_type_id / edge_type_id and token ids
            if "node_type_id" not in g.ndata:
                if "node_token_ids" in g.ndata:
                    g.ndata["node_type_id"] = (g.ndata["node_token_ids"].view(-1) - 1) // 2
                else:
                    raise AssertionError("Missing node_type_id; run preprocessing first")
            if "edge_type_id" not in g.edata:
                if "edge_token_ids" in g.edata:
                    g.edata["edge_type_id"] = (g.edata["edge_token_ids"].view(-1)) // 2
                else:
                    g.edata["edge_type_id"] = torch.zeros(g.num_edges(), dtype=torch.long)
            self._node_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.ndata["node_type_id"].tolist())}
            self._edge_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.edata["edge_type_id"].tolist())}
        self._cache_built = True

    def get_dataset_task_type(self) -> str:
        return "classification"

    def get_num_classes(self) -> int:
        return 2

    def get_loss_config(self) -> Optional[Dict[str, Any]]:
        """Loss config for molhiv. Standard CE by default (Focal/Weighted CE did not improve AUC)."""
        # Standard cross-entropy (default)
        return {
            'method': 'standard'
        }

        # Alternative: Focal Loss (no AUC improvement in testing)
        # return {
        #     'method': 'focal',
        #     'gamma': 2.5,
        #     'alpha': 1.0,
        #     'auto_weights': False
        # }

        # Alternative: Weighted CE (no AUC improvement in testing)
        # return {
        #     'method': 'weighted',
        #     'auto_weights': True
        # }

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
        atomic_num = self.get_node_attribute(graph, node_id)
        return str(atomic_num)

    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        et = self.get_edge_attribute(graph, edge_id)
        return self.BOND_TYPES.get(int(et), str(int(et)))

    def get_most_frequent_edge_type(self) -> str:
        return "SINGLE"

    def get_edge_type_id_by_name(self, name: str) -> int:
        inv = {v: k for k, v in self.BOND_TYPES.items()}
        return int(inv[name])

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
        return [self.BOND_TYPES.get(int(v), str(int(v))) for v in graph.edata["edge_type_id"][ids].tolist()]

    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.ndata["node_type_id"]

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.edata["edge_type_id"]

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.ndata["node_token_ids"]

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.edata["edge_token_ids"]

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        token_map = {}
        # Atoms: odd domain
        for atomic_num in range(1, 119):
            odd_token = 2 * atomic_num + 1
            token_map[("atom", atomic_num)] = odd_token
        # Bonds: even domain
        for bond_type_id, _name in self.BOND_TYPES.items():
            even_token = 2 * int(bond_type_id)
            token_map[("bond", int(bond_type_id))] = even_token
        return token_map


