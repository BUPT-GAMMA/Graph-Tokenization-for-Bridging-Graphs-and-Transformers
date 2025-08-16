from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple
import json

import dgl
import numpy as np
import torch

from .base_loader import BaseDataLoader
from config import ProjectConfig
from utils.logger import get_logger


logger = get_logger(__name__)


class DDLoader(BaseDataLoader):
    """DD 图分类数据集（TU，蛋白质）。"""

    def __init__(self, config: ProjectConfig, dataset_name: str = "DD", target_property: Optional[str] = None):
        super().__init__(dataset_name, config, target_property)
        self._all_data: Optional[List[Dict[str, Any]]] = None
        self._cache_built: bool = False
        self._node_attr_cache: Dict[int, Dict[int, int]] = {}
        self._edge_attr_cache: Dict[int, Dict[int, int]] = {}
        self._normalized_name = "dd"
        self.load_data()

    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"📂 读取 DD 预处理目录: {self.data_dir}")
        train_index_file = self.data_dir / "train_index.json"
        test_index_file = self.data_dir / "test_index.json"
        val_index_file = self.data_dir / "val_index.json"
        for f in (train_index_file, test_index_file, val_index_file):
            if not f.exists():
                raise FileNotFoundError("索引文件不存在，请先运行预处理脚本")
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"统一数据文件不存在: {data_file}")
        if self._all_data is None:
            with open(data_file, "rb") as f:
                raw_data: List[Tuple[dgl.DGLGraph, int]] = pickle.load(f)
            all_data: List[Dict[str, Any]] = []
            for i, (graph, label_int) in enumerate(raw_data):
                sample = {
                    "id": f"{self._normalized_name}_{i}",
                    "dgl_graph": graph,
                    "num_nodes": int(graph.num_nodes()),
                    "num_edges": int(graph.num_edges()),
                    "properties": {"label": int(label_int)},
                    "dataset_name": self.dataset_name,
                    "data_type": "protein_graph",
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
            "dataset_type": "protein_graph",
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
            if "node_token_ids" not in g.ndata:
                raise AssertionError("缺少 node_token_ids，请先运行预处理生成")
            node_token_ids = g.ndata["node_token_ids"].long()
            g.ndata["node_type_id"] = node_token_ids.view(-1)
            if "edge_token_ids" not in g.edata:
                zeros = torch.zeros(g.num_edges(), dtype=torch.long)
                g.edata["edge_token_ids"] = zeros.view(-1, 1)
            g.edata["edge_type_id"] = g.edata["edge_token_ids"].view(-1)
            self._node_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.ndata["node_type_id"].tolist())}
            self._edge_attr_cache[gid] = {int(i): int(v) for i, v in enumerate(g.edata["edge_type_id"].tolist())}
        self._cache_built = True

    def get_dataset_task_type(self) -> str:
        return "classification"

    def get_num_classes(self) -> int:
        return 2

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
        return str(self.get_node_attribute(graph, node_id))

    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return str(self.get_edge_attribute(graph, edge_id))

    def get_most_frequent_edge_type(self) -> str:
        return "0"

    def get_edge_type_id_by_name(self, name: str) -> int:
        return int(name)

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
        return graph.ndata["node_token_ids"]

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return graph.edata["edge_token_ids"]

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        return {}