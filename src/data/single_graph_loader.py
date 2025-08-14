#!/usr/bin/env python3
"""
单图数据集加载器
==============

用于加载经典的图机器学习数据集（Cora、Citeseer、PubMed等）
这些数据集通常包含一个大图，与分子数据集的多个小图不同
"""

import os
import pickle
import dgl
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class SingleGraphLoader:
    """单图数据集加载器"""
    
    def __init__(self, data_root: str = "data/raw/small"):
        """
        初始化单图数据集加载器
        
        Args:
            data_root: 数据集根目录
        """
        self.data_root = Path(data_root)
        self.available_datasets = self._scan_datasets()
        
    def _scan_datasets(self) -> List[str]:
        """扫描可用的数据集"""
        datasets = []
        if self.data_root.exists():
            for item in self.data_root.iterdir():
                if item.is_dir() and (item / "graph.bin").exists():
                    datasets.append(item.name)
        return sorted(datasets)
    
    def list_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        return self.available_datasets.copy()
    
    def load_graph(self, dataset_name: str) -> Tuple[dgl.DGLGraph, Dict[str, Any]]:
        """
        加载指定数据集的图
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            (graph, metadata): DGL图和元数据
        """
        if dataset_name not in self.available_datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在。可用数据集: {self.available_datasets}")
        
        dataset_path = self.data_root / dataset_name
        
        # 加载DGL图
        graph_path = dataset_path / "graph.bin"
        graphs, _ = dgl.load_graphs(str(graph_path))
        graph = graphs[0]  # 只有一个图
        
        # 加载节点名称映射（如果存在）
        node_names = {}
        node_name_path = dataset_path / "graph_node_name.pkl"
        if node_name_path.exists():
            with open(node_name_path, 'rb') as f:
                node_names = pickle.load(f)
        
        # 收集图的元数据
        metadata = self._extract_metadata(graph, node_names, dataset_name)
        
        return graph, metadata
    
    def _extract_metadata(self, graph: dgl.DGLGraph, node_names: Dict, dataset_name: str) -> Dict[str, Any]:
        """提取图的元数据"""
        
        metadata = {
            'dataset_name': dataset_name,
            'num_nodes': graph.num_nodes(),
            'num_edges': graph.num_edges(),
            'node_types': list(graph.ntypes) if hasattr(graph, 'ntypes') else ['default'],
            'edge_types': list(graph.etypes) if hasattr(graph, 'etypes') else ['default'],
            'is_heterograph': len(graph.ntypes) > 1 if hasattr(graph, 'ntypes') else False,
            'node_names': node_names,
            'node_features': {},
            'edge_features': {}
        }
        
        # 检查节点特征
        if hasattr(graph, 'ndata'):
            for key, data in graph.ndata.items():
                try:
                    if hasattr(data, 'shape'):
                        shape = list(data.shape)
                        dtype = str(data.dtype)
                    else:
                        # 处理非tensor数据
                        shape = [len(data)] if hasattr(data, '__len__') else [1]
                        dtype = str(type(data))
                    
                    metadata['node_features'][key] = {
                        'shape': shape,
                        'dtype': dtype,
                        'description': self._get_feature_description(key)
                    }
                except Exception as e:
                    metadata['node_features'][key] = {
                        'shape': 'unknown',
                        'dtype': 'unknown',
                        'description': f'特征解析失败: {str(e)}'
                    }
        
        # 检查边特征  
        if hasattr(graph, 'edata'):
            for key, data in graph.edata.items():
                try:
                    if hasattr(data, 'shape'):
                        shape = list(data.shape)
                        dtype = str(data.dtype)
                    else:
                        # 处理非tensor数据
                        shape = [len(data)] if hasattr(data, '__len__') else [1]
                        dtype = str(type(data))
                    
                    metadata['edge_features'][key] = {
                        'shape': shape,
                        'dtype': dtype,
                        'description': self._get_feature_description(key)
                    }
                except Exception as e:
                    metadata['edge_features'][key] = {
                        'shape': 'unknown',
                        'dtype': 'unknown',
                        'description': f'特征解析失败: {str(e)}'
                    }
        
        return metadata
    
    def _get_feature_description(self, feature_name: str) -> str:
        """获取特征描述"""
        descriptions = {
            'feat': '节点特征向量',
            'label': '节点标签',
            'train_mask': '训练集掩码',
            'val_mask': '验证集掩码',
            'test_mask': '测试集掩码'
        }
        return descriptions.get(feature_name, f'特征: {feature_name}')
    
    def create_full_graph_samples(self, graph: dgl.DGLGraph, num_samples: int = 10, 
                                 strategy: str = 'random_start') -> List[Dict[str, Any]]:
        """
        从单个大图创建多个完整图样本，用于序列化
        
        Args:
            graph: DGL图
            num_samples: 采样次数
            strategy: 采样策略 ('random_start', 'multi_method', 'diverse')
            
        Returns:
            完整图样本列表，每个样本包含DGL图和相关信息
        """
        
        samples = []
        total_nodes = graph.num_nodes()
        
        logger.info(f"图统计: {total_nodes}个节点, {graph.num_edges()}条边")
        
        if len(graph.ntypes) > 1:
            logger.info(f"异构图: {len(graph.ntypes)}种节点类型, {len(graph.etypes)}种边类型")
            logger.debug(f"节点分布: {[(ntype, graph.num_nodes(ntype)) for ntype in graph.ntypes]}")
        
        for i in range(num_samples):
            # 创建样本数据结构，兼容现有的序列化接口
            sample = {
                'dgl_graph': graph,  # 使用完整图
                'graph_id': f'full_graph_sample_{i}',
                'num_nodes': total_nodes,
                'num_edges': graph.num_edges(),
                'sample_id': i,
                'metadata': {
                    'is_full_graph': True,
                    'is_heterograph': len(graph.ntypes) > 1,
                    'node_types': list(graph.ntypes),
                    'edge_types': list(graph.etypes),
                    'sample_strategy': strategy,
                    'sample_variation': self._get_sample_variation(i, strategy, graph)
                }
            }
            
            samples.append(sample)
        
        return samples
    
    def _get_sample_variation(self, sample_id: int, strategy: str, graph: dgl.DGLGraph) -> Dict[str, Any]:
        """
        获取不同样本的变化策略，为同一个图产生不同的序列化结果
        
        Args:
            sample_id: 样本ID
            strategy: 采样策略
            graph: DGL图
            
        Returns:
            样本变化信息
        """
        
        variation = {
            'sample_id': sample_id,
            'random_seed': sample_id * 42,  # 为每个样本设置不同的随机种子
        }
        
        if strategy == 'random_start':
            # 随机起始节点策略
            if len(graph.ntypes) == 1:
                # 同构图：随机选择起始节点
                start_node = sample_id % graph.num_nodes()
                variation.update({
                    'start_node': start_node,
                    'start_node_type': 'default'
                })
            else:
                # 异构图：轮换不同类型的起始节点
                node_types = list(graph.ntypes)
                start_type = node_types[sample_id % len(node_types)]
                start_node = sample_id % graph.num_nodes(start_type)
                variation.update({
                    'start_node': start_node,
                    'start_node_type': start_type
                })
        
        elif strategy == 'multi_method':
            # 多方法策略：模拟不同的遍历方法
            methods = ['dfs', 'bfs', 'random_walk', 'topo']
            variation.update({
                'traversal_method': methods[sample_id % len(methods)],
                'method_params': {'depth_limit': 10 + sample_id}
            })
        
        elif strategy == 'diverse':
            # 多样化策略：结合随机起始和方法变化
            node_types = list(graph.ntypes) if len(graph.ntypes) > 1 else ['default']
            start_type = node_types[sample_id % len(node_types)]
            
            if start_type != 'default':
                start_node = sample_id % graph.num_nodes(start_type)
            else:
                start_node = sample_id % graph.num_nodes()
            
            traversal_methods = ['depth_first', 'breadth_first', 'random']
            method = traversal_methods[sample_id % len(traversal_methods)]
            
            variation.update({
                'start_node': start_node,
                'start_node_type': start_type,
                'traversal_method': method,
                'exploration_factor': 0.1 + (sample_id % 5) * 0.2
            })
        
        return variation
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集的详细信息"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在")
        
        try:
            graph, metadata = self.load_graph(dataset_name)
            
            info = {
                'name': dataset_name,
                'description': self._get_dataset_description(dataset_name),
                'statistics': {
                    'nodes': metadata['num_nodes'],
                    'edges': metadata['num_edges'],
                    'node_types': len(metadata['node_types']),
                    'edge_types': len(metadata['edge_types']),
                    'is_heterograph': metadata['is_heterograph']
                },
                'features': {
                    'node_features': list(metadata['node_features'].keys()),
                    'edge_features': list(metadata['edge_features'].keys())
                },
                'suitable_for': self._get_suitable_tasks(dataset_name)
            }
            
            return info
            
        except Exception as e:
            return {
                'name': dataset_name,
                'error': f'加载失败: {str(e)}',
                'available': False
            }
    
    def _get_dataset_description(self, dataset_name: str) -> str:
        """获取数据集描述"""
        descriptions = {
            'cora': 'Cora引文网络 - 机器学习论文引用关系',
            'citeseer': 'CiteSeer引文网络 - 计算机科学论文引用关系', 
            'pubmed': 'PubMed引文网络 - 生物医学论文引用关系',
            'dblp': 'DBLP异构网络 - 作者、论文、会议、术语关系',
            'imdb': 'IMDB网络 - 电影相关实体关系',
            'yelp': 'Yelp网络 - 用户、商家、评论关系',
            'lastfm': 'Last.FM网络 - 音乐社交网络关系'
        }
        return descriptions.get(dataset_name, f'{dataset_name}数据集')
    
    def _get_suitable_tasks(self, dataset_name: str) -> List[str]:
        """获取数据集适合的任务类型"""
        # 根据数据集特点推荐适合的任务
        tasks = ['图序列化', 'BPE压缩', '图结构分析']
        
        if dataset_name in ['cora', 'citeseer', 'pubmed']:
            tasks.extend(['节点分类', '引文分析'])
        elif dataset_name == 'dblp':
            tasks.extend(['异构图分析', '学术网络分析'])
        elif dataset_name == 'imdb':
            tasks.extend(['推荐系统', '实体关系分析'])
        elif dataset_name in ['yelp', 'lastfm']:
            tasks.extend(['社交网络分析', '推荐系统'])
        
        return tasks

# 用于向后兼容的函数
def load_single_graph_dataset(dataset_name: str, data_root: str = "data/raw/small", 
                             num_samples: int = 10, strategy: str = 'random_start') -> List[Dict[str, Any]]:
    """
    加载单图数据集并创建多个完整图样本
    
    Args:
        dataset_name: 数据集名称
        data_root: 数据根目录
        num_samples: 采样次数
        strategy: 采样策略 ('random_start', 'multi_method', 'diverse')
        
    Returns:
        完整图样本列表，兼容现有的序列化接口
    """
    loader = SingleGraphLoader(data_root)
    graph, metadata = loader.load_graph(dataset_name)
    samples = loader.create_full_graph_samples(graph, num_samples, strategy)
    
    # 为每个样本添加数据集元数据
    for sample in samples:
        sample['dataset_metadata'] = metadata
    
    return samples 