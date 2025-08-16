"""
MNIST超像素图数据加载器

基于MNIST超像素图数据集实现，符合BaseDataLoader接口规范。
支持将3维节点特征和1维边特征映射为紧凑的token序列。
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
    """
    MNIST超像素图数据加载器
    
    节点特征格式: [pixel_id, y_id, x_id]
    - pixel_id: 0-255 (像素值分箱)
    - y_id: 0-27 (Y坐标分箱)
    - x_id: 0-27 (X坐标分箱)
    
    边特征格式: [distance_id]
    - distance_id: 0-39 (距离分箱)
    """
    
    def __init__(self, config, target_property: str = "label"):
        """
        初始化MNIST数据加载器
        
        Args:
            config: 项目配置
            target_property: 目标属性名称，默认"label"
        """
        super().__init__("mnist", config, target_property)
        # Token映射基础值 - 使用通用配置
        self.PIXEL_TOKEN_BASE = 0
        self.Y_COORD_TOKEN_BASE = 256
        self.X_COORD_TOKEN_BASE = 284
        self.DISTANCE_TOKEN_BASE = 314
        
        # 使用全局的节点起止token
        self.NODE_START_TOKEN = self.config.node_start_token_id  # 节点开始token
        self.NODE_END_TOKEN = self.config.node_end_token_id      # 节点结束token
        
        # 最大token值 (考虑新的特殊token)
        self.max_token_value = max(353, self.NODE_END_TOKEN)

    def _load_processed_data(self) -> Tuple[List, List, List]:
        """
        加载处理后的MNIST数据
        
        Returns:
            Tuple[List, List, List]: (train_data, val_data, test_data)
        """
        data_dir = self.data_dir
        
        # 加载主要数据文件
        data_file = os.path.join(data_dir, "data.pkl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"MNIST数据文件不存在: {data_file}")
            
        all_data = self._load_data_file(data_file)
        
        # 加载索引文件
        train_index_file = os.path.join(data_dir, "train_index.json")
        val_index_file = os.path.join(data_dir, "val_index.json")
        test_index_file = os.path.join(data_dir, "test_index.json")
        
        with open(train_index_file, 'r') as f:
            train_indices = json.load(f)
        with open(val_index_file, 'r') as f:
            val_indices = json.load(f)
        with open(test_index_file, 'r') as f:
            test_indices = json.load(f)
        
        # 根据索引划分数据
        train_data = [all_data[i] for i in train_indices]
        val_data = [all_data[i] for i in val_indices]
        test_data = [all_data[i] for i in test_indices]
        
        return train_data, val_data, test_data
      
    def _load_data_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载单个数据文件"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)

            # 转换tuple格式为dict格式
            data = []
            for i, (graph, label) in enumerate(raw_data):
                sample = {
                    'id': f"image_{i}",
                    'dgl_graph': graph,
                    'num_nodes': graph.num_nodes(),
                    'num_edges': graph.num_edges() // 2,  # DGL图是双向的
                    'properties': {'label': label},
                    'dataset_name': self.dataset_name,
                    'data_type': 'image_graph'
                }
                data.append(sample)
            
            return data
        except Exception as e:
            logger.error(f"❌ 加载数据文件失败 {file_path}: {e}")
            raise
        
    def _extract_labels(self, data: List[Tuple[Any, Any]]) -> List[int]:
        """
        从数据中提取标签
        
        Args:
            data: List[(graph, label)]
            
        Returns:
            List[int]: 标签列表 (0-9)
        """
        return [sample['properties']['label'] for sample in data]
        
    def _get_data_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息
        
        Returns:
            元信息字典
        """
        # 确保数据已加载
        if self._train_data is None:
            self.load_data()
        
        assert self._all_data is not None
        all_data = self._all_data
        
        # 统计信息
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

    # ---------------- 下游任务元信息（供上层自动推断） ----------------
    def get_dataset_task_type(self) -> str:
        """MNIST 只能做分类任务。"""
        return "classification"

    def get_num_classes(self) -> int:
        """MNIST 固定为 10 类。"""
        return 10

    def get_default_target_property(self) -> str:
        """分类标签字段名。"""
        return "label"
    
    def get_downstream_label_keys(self) -> List[str]:
        """返回数据集支持的所有标签属性名列表。"""
        return ["label"]  # MNIST只有一个分类标签
        
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """
        获取节点的token列表
        
        Args:
            graph: DGL图对象
            node_id: 节点ID
            ntype: 节点类型 (可选，MNIST中忽略)
            
        Returns:
            List[int]: 节点token列表 [start, pixel, y, x, end]
        """
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
        """
        获取边的token列表
        
        Args:
            graph: DGL图对象
            edge_id: 边ID
            etype: 边类型 (可选，MNIST中忽略)
            
        Returns:
            List[int]: 边token列表 [distance]
        """
        distance = int(graph.edata['feature'][edge_id].item())
        return [self.DISTANCE_TOKEN_BASE + distance]
        
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        """
        获取节点的关键属性值（用于token映射）
        
        Args:
            graph: DGL图对象
            node_id: 节点ID
            
        Returns:
            int: 节点的关键属性值 (像素值)
        """
        return int(graph.ndata['feature'][node_id][0].item())
        
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        """
        获取边的关键属性值（用于token映射）
        
        Args:
            graph: DGL图对象
            edge_id: 边ID
            
        Returns:
            int: 边的关键属性值 (距离)
        """
        return int(graph.edata['feature'][edge_id].item())
        
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        """
        获取节点的类型
        
        Args:
            graph: DGL图对象
            node_id: 节点ID
            
        Returns:
            str: 节点类型 (MNIST中所有节点都是"pixel"类型)
        """
        return "pixel"
        
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        """
        获取边的类型
        
        Args:
            graph: DGL图对象
            edge_id: 边ID
            
        Returns:
            str: 边类型 (MNIST中所有边都是"distance"类型)
        """
        return "distance"
    
    def get_most_frequent_edge_type(self) -> str:
        """
        获取最频繁的边类型
        """
        return "distance"

    def get_edge_type_id_by_name(self, name: str) -> int:
        """
        根据边类型名称获取其数值ID。
        MNIST中仅存在一种边类型 'distance'，统一返回1；否则抛出异常。
        """
        if name == "distance":
            return 1
        raise AssertionError(f"未知边类型: {name}")
        
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        """
        获取完整的token映射表
        
        Returns:
            Dict[Tuple[str, int], int]: token映射表，键为(类型, 值)元组
        """
        token_map: Dict[Tuple[str, int], int] = {}
        # 节点三种子特征
        for v in range(256):
            token_map[("pixel", v)] = self.PIXEL_TOKEN_BASE + v
        for v in range(28):
            token_map[("y", v)] = self.Y_COORD_TOKEN_BASE + v
        for v in range(28):
            token_map[("x", v)] = self.X_COORD_TOKEN_BASE + v
        # 边距离
        for v in range(40):
            token_map[("distance", v)] = self.DISTANCE_TOKEN_BASE + v
        # 特殊token
        token_map[("special", self.NODE_START_TOKEN)] = self.NODE_START_TOKEN
        token_map[("special", self.NODE_END_TOKEN)] = self.NODE_END_TOKEN
        return token_map

    # ==================== 批量API实现 ====================
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        assert 'feature' in graph.ndata, "缺少节点特征 'feature'"
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
        assert 'feature' in graph.edata, "缺少边特征 'feature'"
        ids = torch.as_tensor(list(edge_ids), dtype=torch.long)
        dist = graph.edata['feature'][ids].long()  # [K]
        tok = (self.DISTANCE_TOKEN_BASE + dist).view(-1, 1)
        return tok.tolist()

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[str]:
        # MNIST 所有节点类型均为 'pixel'
        return ["pixel"] * len(list(node_ids))

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[str]:
        # MNIST 所有边类型均为 'distance'
        return ["distance"] * len(list(edge_ids))

    # ==================== 整图张量接口实现 ====================
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # 仅一种类型，统一返回1
        return torch.ones(graph.num_nodes(), dtype=torch.long)

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        # 仅一种类型，统一返回1
        return torch.ones(graph.num_edges(), dtype=torch.long)

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'feature' in graph.ndata, "缺少节点特征 'feature'"
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
        assert 'feature' in graph.edata, "缺少边特征 'feature'"
        dist = graph.edata['feature'].long()  # [E]
        tok = (self.DISTANCE_TOKEN_BASE + dist).view(-1, 1)
        return tok.long()

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()
        