"""MNIST superpixel graph data loader.

Maps 3-dim node features and 1-dim edge features to compact token sequences.
"""

import os
import json
from pathlib import Path
import pickle
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Sequence
import dgl
import torch

from ..base_loader import BaseDataLoader
from utils.logger import get_logger
logger = get_logger(__name__)

class MNISTDataLoader(BaseDataLoader):
    """MNIST superpixel graph data loader (classification, 10 classes).

    Node features: [pixel_id(0-255), y_id(0-27), x_id(0-27)].
    Edge features: [distance_id(0-39)].
    """
    
    def __init__(self, config, target_property: str = "label"):
        super().__init__("mnist", config, target_property)
        # Token mapping base values
        self.PIXEL_TOKEN_BASE = 0
        self.Y_COORD_TOKEN_BASE = 256
        self.X_COORD_TOKEN_BASE = 284
        self.DISTANCE_TOKEN_BASE = 314
        
        # Global node start/end tokens
        self.NODE_START_TOKEN = self.config.node_start_token_id
        self.NODE_END_TOKEN = self.config.node_end_token_id
        
        # Max token value (accounting for special tokens)
        self.max_token_value = max(353, self.NODE_END_TOKEN)

    def _load_processed_data(self) -> Tuple[List, List, List]:
        data_dir = self.data_dir
        
        # Load main data file
        data_file = os.path.join(data_dir, "data.pkl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MNIST data file not found: {data_file}")
            
        all_data = self._load_data_file(data_file)
        
        # Load index files
        train_index_file = os.path.join(data_dir, "train_index.json")
        val_index_file = os.path.join(data_dir, "val_index.json")
        test_index_file = os.path.join(data_dir, "test_index.json")
        
        with open(train_index_file, 'r') as f:
            train_indices = json.load(f)
        with open(val_index_file, 'r') as f:
            val_indices = json.load(f)
        with open(test_index_file, 'r') as f:
            test_indices = json.load(f)
        
        # Split data by indices
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        test_data = [all_data[i] for i in test_indices]
        
        return train_data, val_data, test_data
      
    def _load_data_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single data file."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)

            # Convert tuple format to dict format
            data = []
            for i, (graph, label) in enumerate(raw_data):
                sample = {
                    'id': f"image_{i}",
                    'dgl_graph': graph,
                    'num_nodes': graph.num_nodes(),
                    'num_edges': graph.num_edges() // 2,  # DGL stores bidirectional edges
                    'properties': {'label': label},
                    'dataset_name': self.dataset_name,
                    'data_type': 'image_graph'
                }
                data.append(sample)
            
            return data
        except Exception as e:
            logger.error(f"Failed to load data file {file_path}: {e}")
            raise
        
    def _extract_labels(self, data: List[Tuple[Any, Any]]) -> List[int]:
        return [sample['properties']['label'] for sample in data]
        
    def _get_data_metadata(self) -> Dict[str, Any]:
        # Ensure data is loaded
        if self._train_data is None:
            self.load_data()
        
        assert self._all_data is not None
        all_data = self._all_data
        
        # Statistics
        num_samples = len(all_data)
        num_nodes_list = []
        num_edges_list = []
        label_counts: Dict[int, int] = {}
        
        for sample in all_data:
            if 'dgl_graph' in sample and isinstance(sample['dgl_graph'], dgl.DGLGraph):
                graph = sample['dgl_graph']
                num_nodes_list.append(graph.num_nodes())
                num_edges_list.append(graph.num_edges())
            if 'properties' in sample and 'label' in sample['properties']:
                lbl = int(sample['properties']['label'])
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_type': 'image_graph',
            'data_source': 'mnist_superpixel',
            'total_samples': num_samples,
            'num_classes': 10,
            'avg_num_nodes': int(np.mean(num_nodes_list)) if num_nodes_list else 0,
            'avg_num_edges': int(np.mean(num_edges_list)) if num_edges_list else 0,
            'target_property': self.target_property,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': os.path.join(self.config.data_dir, "mnist"),
            'label_distribution': label_counts,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO
            },
            'feature_info': {
                'node_features': ['pixel_id', 'y_id', 'x_id'],
                'edge_features': ['distance_id'],
                'pixel_range': [0, 255],
                'coordinate_range': [0, 27],
                'distance_range': [0, 39]
            }
        }
        
        return metadata

    # ---------------- Downstream task info ----------------
    def get_dataset_task_type(self) -> str:
        """MNIST is classification only."""
        return "classification"

    def get_num_classes(self) -> int:
        """MNIST has 10 classes."""
        return 10

    def get_default_target_property(self) -> str:
        """Classification label field name."""
        return "label"
    
    def get_downstream_label_keys(self) -> List[str]:
        """Return available label keys."""
        return ["label"]
        
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        features = graph.ndata['feature'][node_id]  # [pixel_id, y_id, x_id]
        
        pixel_id = int(features[0].item())
        y_id = int(features[1].item())
        x_id = int(features[2].item())
        
        return [
            self.NODE_START_TOKEN,
            self.PIXEL_TOKEN_BASE + pixel_id,
            self.Y_COORD_TOKEN_BASE + y_id,
            self.X_COORD_TOKEN_BASE + x_id,
            self.NODE_END_TOKEN
        ]
        
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        distance = int(graph.edata['feature'][edge_id].item())
        return [self.DISTANCE_TOKEN_BASE + distance]
        
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        return int(graph.ndata['feature'][node_id][0].item())
        
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        return int(graph.edata['feature'][edge_id].item())
        
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        return "pixel"
        
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return "distance"
    
    def get_most_frequent_edge_type(self) -> str:
        return "distance"

    def get_edge_type_id_by_name(self, name: str) -> int:
        if name == "distance":
            return 1
        raise AssertionError(f"Unknown edge type: {name}")
        
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        token_map: Dict[Tuple[str, int], int] = {}
        # Node sub-features
        for v in range(256):
            token_map[("pixel", v)] = self.PIXEL_TOKEN_BASE + v
        for v in range(28):
            token_map[("y", v)] = self.Y_COORD_TOKEN_BASE + v
        for v in range(28):
            token_map[("x", v)] = self.X_COORD_TOKEN_BASE + v
        # Edge distance
        for v in range(40):
            token_map[("distance", v)] = self.DISTANCE_TOKEN_BASE + v
        # Special tokens
        token_map[("special", self.NODE_START_TOKEN)] = self.NODE_START_TOKEN
        token_map[("special", self.NODE_END_TOKEN)] = self.NODE_END_TOKEN
        return token_map

    # ==================== Bulk API ====================
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.ndata, "Missing node feature 'feature'"
        ids = torch.as_tensor(list(node_ids), dtype=torch.long)
        feats = graph.ndata['feature'][ids].long()  # [K, 3]
        pixel = feats[:, 0]
        # y = feats[:, 1]
        # x = feats[:, 2]
        # start = torch.full_like(pixel, fill_value=int(self.NODE_START_TOKEN))
        # end = torch.full_like(pixel, fill_value=int(self.NODE_END_TOKEN))
        tok = torch.stack([
            # start,
            self.PIXEL_TOKEN_BASE + pixel,
            # self.Y_COORD_TOKEN_BASE + y,
            # self.X_COORD_TOKEN_BASE + x,
            # end
        ], dim=1)
        return tok.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.edata, "Missing edge feature 'feature'"
        ids = torch.as_tensor(list(edge_ids), dtype=torch.long)
        dist = graph.edata['feature'][ids].long()  # [K]
        tok = (self.DISTANCE_TOKEN_BASE + dist).view(-1, 1)
        return tok.tolist()

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[str]:
        # All MNIST nodes are 'pixel'
        return ["pixel"] * len(list(node_ids))

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[str]:
        # All MNIST edges are 'distance'
        return ["distance"] * len(list(edge_ids))

    # ==================== Whole-graph tensor interface ====================
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # Single type, return 1
        return torch.ones(graph.num_nodes(), dtype=torch.long)

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # Single type, return 1
        return torch.ones(graph.num_edges(), dtype=torch.long)

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'feature' in graph.ndata, "Missing node feature 'feature'"
        feats = graph.ndata['feature'].long()  # [N, 3]
        pixel = feats[:, 0]
        # y = feats[:, 1]
        # x = feats[:, 2]
        # start = torch.full_like(pixel, fill_value=int(self.NODE_START_TOKEN))
        # end = torch.full_like(pixel, fill_value=int(self.NODE_END_TOKEN))
        tok = torch.stack([
            # start,
            self.PIXEL_TOKEN_BASE + pixel,
            # self.Y_COORD_TOKEN_BASE + y,
            # self.X_COORD_TOKEN_BASE + x,
            # end
        ], dim=1)  # [N, 5]
        return tok.long()

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'feature' in graph.edata, "Missing edge feature 'feature'"
        dist = graph.edata['feature'].long()  # [E]
        tok = (self.DISTANCE_TOKEN_BASE + dist).view(-1, 1)
        return tok.long()

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()
        