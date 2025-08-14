"""
MNIST-RAW 栅格图数据加载器
=========================

将原始 MNIST 28x28 灰度图转为规则栅格图：
- 节点：每个像素一个节点，节点特征仅包含一个离散数值（0-255 的像素值）
- 边：4-邻接（上下左右）无向图，以有向边的形式存储；边特征统一为常数 1（离散值）

存储格式（data_dir/mnist_raw/data.pkl）：
- 列表，每个元素为 (dgl_graph, label)
- 三个划分索引文件：train_index.json / val_index.json / test_index.json

说明：遵循 BaseDataLoader 统一接口；不包含任何隐式回退逻辑。
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

from .base_loader import BaseDataLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class MNISTRawDataLoader(BaseDataLoader):
    """MNIST 原始栅格图数据加载器（分类任务，10 类）。"""

    def __init__(self, config, target_property: str = "label"):
        super().__init__("mnist_raw", config, target_property)

    # ==================== 基础加载与元信息 ====================
    def _load_processed_data(self) -> Tuple[List, List, List]:
        data_dir = self.data_dir

        data_file = os.path.join(data_dir, "data.pkl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MNIST-RAW 数据文件不存在: {data_file}")

        # 加载轻量数据：仅像素与标签；按索引构图，避免一次性构建所有图
        raw_pairs = self._load_raw_pairs(Path(data_file))  # list of (np.ndarray[28,28], int)

        # 加载划分索引
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
                # 严格不变量检查
                expected_undirected_edges = (W - 1) * H + (H - 1) * W
                assert g.num_nodes() == H * W, f"节点数不匹配: {g.num_nodes()} != {H*W}"
                assert g.num_edges() == 2 * expected_undirected_edges, (
                    f"边数不匹配: {g.num_edges()} != {2*expected_undirected_edges}"
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

        # 缓存 all_data 以满足 BaseDataLoader 的统计接口
        self._all_data = train_data + val_data + test_data

        return train_data, val_data, test_data

    def _load_raw_pairs(self, file_path: Path) -> List[Tuple[np.ndarray, int]]:
        """读取轻量 data.pkl，返回 (image_uint8[28,28], label) 列表。"""
        try:
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
            pairs: List[Tuple[np.ndarray, int]] = []
            for item in raw:
                img, lbl = item
                if isinstance(img, np.ndarray):
                    arr = img.astype(np.uint8)
                else:
                    # 安全起见，尝试转换
                    arr = np.array(img, dtype=np.uint8)
                pairs.append((arr, int(lbl)))
            return pairs
        except Exception as e:
            logger.error(f"❌ 读取轻量数据失败 {file_path}: {e}")
            raise

    # -------- 图构建 --------
    def _image_to_dgl(self, img_uint8: np.ndarray) -> dgl.DGLGraph:
        H, W = img_uint8.shape
        assert H == 28 and W == 28, f"期望 28x28 图像，得到 {H}x{W}"

        # 预构建并缓存规则网格边
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
        # 节点特征：像素值 [N,1]
        pix = torch.from_numpy(img_uint8.reshape(-1, 1))  # uint8
        g.ndata['feature'] = pix
        # 边特征：常数 1 [E]
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

    # ---------------- 下游任务元信息 ----------------
    def get_dataset_task_type(self) -> str:
        return "classification"

    def get_num_classes(self) -> int:
        return 10

    def get_default_target_property(self) -> str:
        return "label"

    def get_downstream_label_keys(self) -> List[str]:
        return ["label"]

    # ==================== Token/属性接口 ====================
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        # ndata['feature'] 形状 [N, 1]，dtype=uint8
        return int(graph.ndata['feature'][node_id, 0].item())

    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        # edata['feature'] 形状 [E]，统一常数 1
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
        raise AssertionError(f"未知边类型: {name}")

    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        # 简单 token：像素值本身作为 token（0..255）
        return [self.get_node_attribute(graph, node_id)]

    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        # 简单 token：边常数 1
        return [self.get_edge_attribute(graph, edge_id)]

    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        token_map: Dict[Tuple[str, int], int] = {}
        for v in range(256):
            token_map[("pixel", v)] = v
        token_map[("edge", 1)] = 1
        return token_map

    # ==================== 批量/整图张量接口 ====================
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.ndata, "缺少节点特征 'feature'"
        ids = torch.as_tensor(list(node_ids), dtype=torch.long)
        pix = graph.ndata['feature'][ids, 0].to(torch.long).view(-1, 1)
        return pix.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.edata, "缺少边特征 'feature'"
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
        # 返回 [N, 1] LongTensor
        pix = graph.ndata['feature'][:, 0].to(torch.long).view(-1, 1)
        return pix

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # 返回 [E, 1] LongTensor
        val = graph.edata['feature'].to(torch.long).view(-1, 1)
        return val

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()


