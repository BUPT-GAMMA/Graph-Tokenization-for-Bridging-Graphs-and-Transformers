"""
QM9Test数据加载器
================

专门用于测试的小规模QM9数据加载器，使用10%的QM9数据。
继承自QM9Loader，但使用qm9test目录下的数据。
"""

import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import json
from pathlib import Path

# 必需依赖
import dgl
import torch

from .qm9_loader import QM9Loader
from config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class QM9TestLoader(QM9Loader):
    """QM9Test数据加载器 - 使用10%的QM9数据"""
    
    def __init__(self, config: ProjectConfig, target_property: Optional[str] = None):
        """
        初始化QM9Test加载器
        
        Args:
            config: 项目配置
            target_property: 目标属性（对于多标签数据集，None表示返回所有属性）
        """
        # 覆盖数据集名称，然后调用父类初始化
        self.dataset_name = "qm9test"
        super().__init__(config, self.dataset_name, target_property)
        
        # 使用配置中的数据根目录，避免依赖当前工作目录
        # BaseDataLoader 已将 data_dir 设为 config.data_dir/dataset_name
        if not self.data_dir.exists():
            raise FileNotFoundError(f"QM9Test数据集目录不存在: {self.data_dir}")
        logger.info(f"🔧 初始化QM9Test数据加载器: {self.data_dir}")
    
    def _get_data_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息
        
        Returns:
            元信息字典
        """
        # 确保数据已加载
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        # 统计信息
        num_samples = len(all_data)
        num_nodes_list = [sample.get('num_nodes', 0) for sample in all_data]
        num_edges_list = [sample.get('num_edges', 0) for sample in all_data]
        
        # 属性统计
        property_stats = {}
        if all_data and 'properties' in all_data[0]:
            properties = all_data[0]['properties']
            for prop_name in self.QM9_PROPERTIES:
                if prop_name in properties:
                    prop_values = [sample.get('properties', {}).get(prop_name) for sample in all_data 
                                 if prop_name in sample.get('properties', {})]
                    if prop_values:
                        property_stats[prop_name] = {
                            'min': float(np.min(prop_values)),
                            'max': float(np.max(prop_values)),
                            'mean': float(np.mean(prop_values)),
                            'std': float(np.std(prop_values))
                        }
        
        # 加载QM9Test特定的元数据
        metadata_file = self.data_dir / "metadata.json"
        qm9test_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    qm9test_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"无法加载QM9Test元数据: {e}")
        
        metadata = {
            'dataset_name': 'qm9test',
            'dataset_type': 'molecular_graph',
            'data_source': 'qm9_subset',
            'total_molecules': num_samples,
            'target_property': self.target_property,
            'qm9_properties': self.QM9_PROPERTIES,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': str(self.data_dir),
            'property_availability': property_stats,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO
            },
            # QM9Test特定信息
            'source_dataset': qm9test_metadata.get('source_dataset', 'qm9'),
            'test_ratio': qm9test_metadata.get('test_ratio', 0.1),
            'original_indices': qm9test_metadata.get('original_indices', []),
            'creation_time': qm9test_metadata.get('creation_time', 'Unknown'),
            'random_state': qm9test_metadata.get('random_state', 42)
        }
        
        return metadata

