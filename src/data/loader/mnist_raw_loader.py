"""MNIST-RAW grid-graph data loader.

Converts raw MNIST 28x28 grayscale images into regular grid graphs:
- Nodes: one per pixel, feature is the discrete pixel value (0-255).
- Edges: 4-connected (up/down/left/right) undirected, stored as directed pairs;
  edge feature is constant 1.
"""

from __future__ import annotations

import os
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import dgl
import torch
import numpy as np

from ..base_loader import BaseDataLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class MNISTRawDataLoader(BaseDataLoader):
    """MNIST raw grid-graph data loader (classification, 10 classes)."""

    def __init__(self, config, target_property: str = "label"):
        super().__init__("mnist_raw", config, target_property)

    # ==================== Loading & metadata ====================
    def _load_processed_data(self) -> Tuple[List, List, List]:
        data_dir = self.data_dir

        data_file = os.path.join(data_dir, "data.pkl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MNIST-RAW data file not found: {data_file}")

        # Load lightweight data: pixels and labels only; build graphs on demand
        raw_pairs = self._load_raw_pairs(Path(data_file))  # list of (np.ndarray[28,28], int)

        # Load split indices
        train_index_file = os.path.join(data_dir, "train_index.json")
        val_index_file = os.path.join(data_dir, "val_index.json")
        test_index_file = os.path.join(data_dir, "test_index.json")

        with open(train_index_file, 'r') as f:
            train_indices = json.load(f)
        with open(val_index_file, 'r') as f:
            val_indices = json.load(f)
        with open(test_index_file, 'r') as f:
            test_indices = json.load(f)

        def build_samples(indices: List[int]) -> List[Dict[str, Any]]:
            samples: List[Dict[str, Any]] = []
            for i in indices:
                img, lbl = raw_pairs[i]
                g = self._image_to_dgl(img)
                H, W = int(img.shape[0]), int(img.shape[1])
                # Invariant check
                expected_undirected_edges = (W - 1) * H + (H - 1) * W
                assert g.num_nodes() == H * W, f"Node count mismatch: {g.num_nodes()} != {H*W}"
                assert g.num_edges() == 2 * expected_undirected_edges, (
                    f"Edge count mismatch: {g.num_edges()} != {2*expected_undirected_edges}"
                )
                sample = {
                    'id': f"image_{i}",
                    'dgl_graph': g,
                    'image_shape': (H, W, 1),
                    'num_nodes': g.num_nodes(),
                    'num_edges': g.num_edges() // 2,
                    'properties': {'label': int(lbl)},
                    'dataset_name': self.dataset_name,
                    'data_type': 'image_grid_graph',
                }
                samples.append(sample)
            return samples

        train_data = build_samples(train_indices)
        val_data = build_samples(val_indices)
        test_data = build_samples(test_indices)

        # Cache all_data for BaseDataLoader statistics
        self._all_data = train_data + val_data + test_data

        return train_data, val_data, test_data

    def _load_raw_pairs(self, file_path: Path) -> List[Tuple[np.ndarray, int]]:
        """Read lightweight data.pkl, return list of (image_uint8[28,28], label)."""
        try:
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
            pairs: List[Tuple[np.ndarray, int]] = []
            for item in raw:
                img, lbl = item
                if isinstance(img, np.ndarray):
                    arr = img.astype(np.uint8)
                else:
                    # Safety fallback
                    arr = np.array(img, dtype=np.uint8)
                pairs.append((arr, int(lbl)))
            return pairs
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise

    # -------- Graph construction --------
    def _image_to_dgl(self, img_uint8: np.ndarray) -> dgl.DGLGraph:
        H, W = img_uint8.shape
        assert H == 28 and W == 28, f"Expected 28x28 image, got {H}x{W}"

        # Pre-build and cache regular grid edges
        if not hasattr(self, "_grid_edges"):
            u_list: List[int] = []
            v_list: List[int] = []
            def node_id(y: int, x: int) -> int:
                return y * W + x
            for y in range(H):
                for x in range(W):
                    nid = node_id(y, x)
                    if x + 1 < W:
                        right = node_id(y, x + 1)
                        u_list.extend([nid, right])
                        v_list.extend([right, nid])
                    if y + 1 < H:
                        down = node_id(y + 1, x)
                        u_list.extend([nid, down])
                        v_list.extend([down, nid])
            self._grid_edges = (
                torch.tensor(u_list, dtype=torch.long),
                torch.tensor(v_list, dtype=torch.long),
            )

        u, v = self._grid_edges
        g = dgl.graph((u, v), num_nodes=H * W)
        # Node feature: pixel value [N,1]
        pix = torch.from_numpy(img_uint8.reshape(-1, 1))  # uint8
        g.ndata['feature'] = pix
        # Edge feature: constant 1 [E]
        g.edata['feature'] = torch.ones(g.num_edges(), dtype=torch.long)
        return g

    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[int]:
        return [int(sample['properties']['label']) for sample in data]

    def _get_data_metadata(self) -> Dict[str, Any]:
        if self._train_data is None:
            self.load_data()
        assert self._all_data is not None
        all_data = self._all_data

        num_samples = len(all_data)
        num_nodes_list = []
        num_edges_list = []
        label_counts: Dict[int, int] = {}

        for sample in all_data:
            g = sample.get('dgl_graph')
            if isinstance(g, dgl.DGLGraph):
                num_nodes_list.append(g.num_nodes())
                num_edges_list.append(g.num_edges())
            lbl = int(sample['properties']['label'])
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_type': 'image_grid_graph',
            'data_source': 'mnist_raw',
            'total_samples': num_samples,
            'num_classes': 10,
            'avg_num_nodes': int(np.mean(num_nodes_list)) if num_nodes_list else 0,
            'avg_num_edges': int(np.mean(num_edges_list)) if num_edges_list else 0,
            'target_property': self.target_property,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': os.path.join(self.config.data_dir, "mnist_raw"),
            'label_distribution': label_counts,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO,
            },
            'feature_info': {
                'node_features': ['pixel'],
                'edge_features': ['link_const_1'],
                'pixel_range': [0, 255],
            },
        }
        return metadata

    # ---------------- Downstream task info ----------------
    def get_dataset_task_type(self) -> str:
        return "classification"

    def get_num_classes(self) -> int:
        return 10

    def get_default_target_property(self) -> str:
        return "label"

    def get_downstream_label_keys(self) -> List[str]:
        return ["label"]

    # ==================== Token / attribute interface ====================
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        # ndata['feature'] shape [N, 1], dtype=uint8
        return int(graph.ndata['feature'][node_id, 0].item())

    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        # edata['feature'] shape [E], constant 1
        return int(graph.edata['feature'][edge_id].item())

    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        return "pixel"

    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return "edge"

    def get_most_frequent_edge_type(self) -> str:
        return "edge"

    def get_edge_type_id_by_name(self, name: str) -> int:
        if name == "edge":
            return 1
        raise AssertionError(f"Unknown edge type: {name}")

    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        # Simple token: pixel value itself (0..255)
        return [self.get_node_attribute(graph, node_id)]

    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        # Simple token: edge constant 1
        return [self.get_edge_attribute(graph, edge_id)]

    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        token_map: Dict[Tuple[str, int], int] = {}
        for v in range(256):
            token_map[("pixel", v)] = v
        token_map[("edge", 1)] = 1
        return token_map

    # ==================== Bulk / whole-graph tensor interface ====================
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.ndata, "Missing node feature 'feature'"
        ids = torch.as_tensor(list(node_ids), dtype=torch.long)
        pix = graph.ndata['feature'][ids, 0].to(torch.long).view(-1, 1)
        return pix.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.edata, "Missing edge feature 'feature'"
        ids = torch.as_tensor(list(edge_ids), dtype=torch.long)
        val = graph.edata['feature'][ids].to(torch.long).view(-1, 1)
        return val.tolist()

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[str]:
        return ["pixel"] * len(list(node_ids))

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[str]:
        return ["edge"] * len(list(edge_ids))

    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return torch.ones(graph.num_nodes(), dtype=torch.long)

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        return torch.ones(graph.num_edges(), dtype=torch.long)

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # Return [N, 1] LongTensor
        pix = graph.ndata['feature'][:, 0].to(torch.long).view(-1, 1)
        return pix

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # Return [E, 1] LongTensor
        val = graph.edata['feature'].to(torch.long).view(-1, 1)
        return val

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()


