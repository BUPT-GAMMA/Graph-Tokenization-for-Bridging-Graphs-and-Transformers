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


class CODE2Loader(BaseDataLoader):
    """ogbg-code2 图到序列数据集（OGB）。

    预处理约定：
    - 节点两维离散特征分别作为两个节点token输出；两者需域不相交，因此第二维加大偏置。
    - 数据集中无边特征，边token统一为0（偶数域）。
    - 写入规范键：`node_token_ids: [N, Dn]`、`edge_token_ids: [E, 1]`，并同步 `feat`。
    """

    def __init__(self, config: ProjectConfig, dataset_name: str = "code2", target_property: Optional[str] = None):
        super().__init__(dataset_name, config, target_property)
        self._all_data: Optional[List[Dict[str, Any]]] = None
        self._cache_built: bool = False
        self._node_attr_cache: Dict[int, Dict[int, List[int]]] = {}
        self._edge_attr_cache: Dict[int, Dict[int, int]] = {}
        self._normalized_name = "code2"
        # 大偏置，确保两个节点token域不重叠，且仍保持奇数域
        self._second_channel_bias = 10_000_000
        self.load_data()

    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"📂 读取 CODE2 预处理目录: {self.data_dir}")
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
                raw_data: List[Tuple[dgl.DGLGraph, Any]] = pickle.load(f)
            all_data: List[Dict[str, Any]] = []
            for i, (graph, label_obj) in enumerate(raw_data):
                sample = {
                    "id": f"{self._normalized_name}_{i}",
                    "dgl_graph": graph,
                    "num_nodes": int(graph.num_nodes()),
                    "num_edges": int(graph.num_edges()),
                    "properties": {"label": label_obj},  # code2 是序列标签
                    "dataset_name": self.dataset_name,
                    "data_type": "program_graph",
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
        return [s.get("properties", {}).get("label") for s in data]

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
            "dataset_type": "program_graph",
            "total_graphs": len(all_data),
            "avg_num_nodes": float(np.mean(num_nodes)),
            "avg_num_edges": float(np.mean(num_edges)),
            "task_type": self.get_dataset_task_type(),
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
            # 已在预处理阶段写回标准键；此处仅构建缓存。
            self._node_attr_cache[gid] = {int(i): g.ndata['node_token_ids'][i].long().tolist() for i in range(g.num_nodes())}
            self._edge_attr_cache[gid] = {int(i): int(g.edata['edge_type_id'][i].item()) for i in range(g.num_edges())}
        self._cache_built = True

    def get_dataset_task_type(self) -> str:
        return "sequence_prediction"

    def get_num_classes(self) -> int:
        return 0

    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        # 返回主通道类型（第一通道还原前的索引）
        tok = graph.ndata["node_token_ids"][int(node_id)]
        return int(((int(tok[0].item())) - 1) // 2)

    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        if graph.num_edges() == 0:
            return 0
        return int(graph.edata["edge_type_id"][int(edge_id)].item())

    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        return str(self.get_node_attribute(graph, node_id))

    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return str(self.get_edge_attribute(graph, edge_id))

    def get_most_frequent_edge_type(self) -> str:
        return "0"

    def get_edge_type_id_by_name(self, name: str) -> int:
        return int(name)

    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        # 返回两通道 token
        t = graph.ndata["node_token_ids"][int(node_id)].long()
        if t.dim() == 0:
            return [int(t.item())]
        return [int(t[0].item()), int(t[1].item())] 

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
        return {}


