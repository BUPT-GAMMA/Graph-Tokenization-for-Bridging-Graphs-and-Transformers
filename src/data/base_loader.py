"""
统一数据加载器基类
================

为所有数据集加载器提供统一的接口和公共逻辑。
设计原则：
1. 统一接口：所有数据集加载器实现相同的接口
2. 简化逻辑：直接从预处理数据读取，移除复杂缓存
3. 配置统一：使用统一的配置管理
4. 多标签支持：支持单标签和多标签模式
5. 数据集划分：由外部三份索引文件（train/val/test）决定，禁止内部随机切分
"""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence
import dgl
import numpy as np
import torch
# 注意：此处不直接使用 tqdm；子类如需要可自行导入使用

from config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseDataLoader(ABC):
    """
    统一数据加载器基类
    
    为所有数据集加载器提供统一的接口和公共逻辑
    """
    
    # 固定的数据集划分比例
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    def __init__(self, dataset_name: str, config: ProjectConfig, 
                 target_property: Optional[str] = None):
        """
        初始化数据加载器
        
        Args:
            dataset_name: 数据集名称
            config: 项目配置
            target_property: 目标属性（对于多标签数据集，None表示返回所有属性）
        """
        self.dataset_name = dataset_name
        self.config = config
        self.target_property = target_property
        
        # 数据存储路径（使用全局配置的 data_dir，避免依赖当前工作目录）
        self.data_dir = Path(self.config.data_dir) / self.dataset_name
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.data_dir}")
        
        # 数据缓存（仅内存）
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._train_labels = None
        self._val_labels = None
        self._test_labels = None
        self._metadata = None
        self.token_map = None
        self.token_readable = None
        self._all_data = None
    @abstractmethod
    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从预处理数据目录加载数据（子类必须实现）
        
        新的数据结构：
        - 统一的数据文件（如data.pkl）
        - 三个index文件（train_index.json, test_index.json, val_index.json）
        - 使用索引来标识不同划分的数据条目
        
        Returns:
            Tuple[List, List, List]: (训练集, 验证集, 测试集)
        """
        pass
    
    @abstractmethod
    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[Any]:
        """
        从数据中提取标签（子类必须实现）
        
        Args:
            data: 数据列表
            
        Returns:
            标签列表（单标签）或标签字典列表（多标签）
        """
        pass
    
    @abstractmethod
    def _get_data_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息（子类必须实现）
        
        Returns:
            元信息字典
        """
        pass

    # ==================== Token管理接口（子类必须实现） ====================
    
    @abstractmethod
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        """
        获取节点的关键属性值（用于token映射）
        
        Args:
            graph: DGL图
            node_id: 节点ID
            
        Returns:
            int: 节点的关键属性值（如原子序数）
        """
        pass
    
    @abstractmethod
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        """
        获取边的关键属性值（用于token映射）
        
        Args:
            graph: DGL图
            edge_id: 边ID
            
        Returns:
            int: 边的关键属性值（如键类型）
        """
        pass
    
    @abstractmethod
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        """
        获取节点的类型
        """
        pass
    
    @abstractmethod
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        """
        获取边的类型
        """
        pass
      
    @abstractmethod
    def get_most_frequent_edge_type(self) -> str:
        """
        获取最频繁的边类型
        """
        pass

    @abstractmethod
    def get_edge_type_id_by_name(self, name: str) -> int:
        """
        根据边类型名称获取其数值ID（用于高效比较与批量处理）。
        """
        pass
      
    @abstractmethod
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """
        获取节点的token列表
        
        Returns:
            List[int]: 节点token列表
        """
        pass
    
    @abstractmethod
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:  
        """
        获取边的token列表
        
        Returns:
            List[int]: 边token列表
        """
        pass
    
    # ==================== 批量API（必须由子类实现；禁止回退） ====================
    
    @abstractmethod
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        """批量获取节点tokens（子类必须实现张量化/高效实现）"""
        pass

    @abstractmethod
    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[List[int]]:
        """批量获取边tokens（子类必须实现张量化/高效实现）"""
        pass

    @abstractmethod
    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[str]:
        """批量获取节点类型（子类必须实现张量化/高效实现）"""
        pass

    @abstractmethod
    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[str]:
        """批量获取边类型（子类必须实现张量化/高效实现）"""
        pass

    # ==================== 整图张量接口（子类必须实现；禁止回退） ====================
    @abstractmethod
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """返回整图节点类型 id，形状 [N] (LongTensor)。"""
        pass

    @abstractmethod
    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """返回整图边类型 id，形状 [E] (LongTensor)。"""
        pass

    @abstractmethod
    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """返回整图节点 token，形状 [N, Dn] (LongTensor)。"""
        pass

    @abstractmethod
    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """返回整图边 token，形状 [E, De] (LongTensor)。"""
        pass

    @abstractmethod
    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回整图边的 (src, dst) 索引（LongTensor）。"""
        pass
      
    
    @abstractmethod
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        """
        获取数据集级别的token映射
        """
        pass
    
    def get_all_data_with_indices(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """
        获取全部数据和对应的划分索引
        
        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, List[int]]]: 
            - 全部数据列表（按原始顺序）
            - 划分索引字典 {'train': [...], 'test': [...], 'val': [...]}
        """
        # 获取划分索引
        split_indices = self.get_split_indices()
        
        if self._all_data is None:
          self.load_data()
        
        return self._all_data, split_indices
    
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                                 List[Any], List[Any], List[Any]]:
        """
        加载数据集（统一接口）
        
        Returns:
            Tuple: (训练数据, 验证数据, 测试数据, 训练标签, 验证标签, 测试标签)
                  标签可以是单标签列表或多标签字典列表
        """
        # 如果已经加载过，直接返回
        if self._train_data is not None:
            return (self._train_data, self._val_data, self._test_data, 
                   self._train_labels, self._val_labels, self._test_labels)
        
        logger.info(f"🔄 开始加载 {self.dataset_name} 数据...")
        logger.info(f"📂 预处理目录: {self.data_dir}")
        
        # 加载预处理数据
        train_data, val_data, test_data = self._load_processed_data()
        logger.info("📄 预处理数据读取完成，开始组装三分数据...")
        
        if not train_data or not val_data or not test_data:
            raise ValueError(f"加载 {self.dataset_name} 数据失败")
        
        logger.info(f"📊 数据加载完成: 训练集{len(train_data)}, 验证集{len(val_data)}, 测试集{len(test_data)}")
        
        # 提取标签
        train_labels = self._extract_labels(train_data)
        val_labels = self._extract_labels(val_data)
        test_labels = self._extract_labels(test_data)
        
        # 缓存数据
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._train_labels = train_labels
        self._val_labels = val_labels
        self._test_labels = test_labels
        
        logger.info(f"✅ 标签提取完成: 目标属性={self.target_property}")
        self._all_data = train_data + val_data + test_data
        
        logger.info("构建数据集token映射...")
        self.token_map = self.get_token_map()
        self.token_readable = {v: k for k, v in self.token_map.items()}
        logger.info("✅ 数据加载器就绪")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
      
    def get_split_indices(self) -> Dict[str, List[int]]:
        """
        获取数据集划分的索引
        
        Returns:
            Dict[str, List[int]]: 包含 'train', 'test', 'val' 键的字典，
                                 每个键对应一个索引列表
        """
        data_dir = self.data_dir
        
        train_index_file = os.path.join(data_dir, "train_index.json")
        val_index_file = os.path.join(data_dir, "val_index.json")
        test_index_file = os.path.join(data_dir, "test_index.json")
        
        try:
            with open(train_index_file, 'r') as f:
                train_indices = json.load(f)
            with open(val_index_file, 'r') as f:
                val_indices = json.load(f)
            with open(test_index_file, 'r') as f:
                test_indices = json.load(f)
            
            return {
                'train': train_indices,
                'test': test_indices,
                'val': val_indices
            }
        except Exception as e:
            # 统一表述，避免数据集名残留带来误导
            raise FileNotFoundError(f"加载索引文件失败: {e}")
    
    def get_smiles_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        获取三个划分的SMILES字符串
        
        Returns:
            Tuple[List[str], List[str], List[str]]: (训练集SMILES, 验证集SMILES, 测试集SMILES)
            
        Raises:
            NotImplementedError: 当数据集不支持SMILES时
        """
        raise NotImplementedError(f"数据集 {self.dataset_name} 不支持SMILES功能")
    
    def get_smiles_by_type(self, smiles_type: str = "1") -> Tuple[List[str], List[str], List[str]]:
        """
        根据类型获取特定格式的SMILES字符串
        
        Args:
            smiles_type: SMILES格式类型 ("1", "2", "3", "4")
                - "1": 直接SMILES
                - "2": 显式氢原子SMILES
                - "3": AddHs SMILES
                - "4": AddHs+显式氢原子SMILES
        
        Returns:
            Tuple[List[str], List[str], List[str]]: (训练集SMILES, 验证集SMILES, 测试集SMILES)
            
        Raises:
            NotImplementedError: 当数据集不支持SMILES时
        """
        raise NotImplementedError(f"数据集 {self.dataset_name} 不支持SMILES功能")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据元信息
        
        Returns:
            元信息字典
        """
        if self._metadata is None:
            self._metadata = self._get_data_metadata()
        
        return self._metadata

    # ---------------- 下游任务元信息（基础默认实现，子类可覆盖） ----------------
    def get_dataset_task_type(self) -> str:
        """返回数据集默认任务类型。默认回归；分类数据集应在子类覆盖。"""
        return "regression"

    def get_num_classes(self) -> int:
        """分类任务的类别数。默认返回1；分类数据集应在子类覆盖为真实类别数。"""
        return 1

    def get_default_target_property(self) -> Optional[str]:
        """回归任务的默认目标属性键。默认返回 None，子类可覆盖。"""
        return None

    def get_downstream_label_keys(self) -> List[str]:
        """返回下游可用的标签键列表。默认空列表，子类可覆盖。"""
        return ['label']

    def get_loss_config(self) -> Optional[Dict[str, Any]]:
        """
        返回数据集推荐的损失函数配置。

        Returns:
            损失配置字典，None表示使用默认配置
            格式: {
                'method': 'focal',  # 'focal' | 'weighted' | 'standard'
                'gamma': 2.5,       # Focal Loss参数
                'alpha': 1.0,       # Focal Loss参数
                'auto_weights': True  # 是否自动计算类别权重
            }
        """
        return None  # 默认无特殊配置
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Returns:
            统计信息字典
        """
        # 确保数据已加载
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        stats = {
            'total_samples': len(all_data),
            'train_samples': len(self._train_data),
            'val_samples': len(self._val_data),
            'test_samples': len(self._test_data),
            'dataset_name': self.dataset_name,
            'target_property': self.target_property,
            'cache_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 图统计信息
        num_nodes_list = []
        num_edges_list = []
        
        for sample in all_data:
            if 'dgl_graph' in sample:
                graph = sample['dgl_graph']
                num_nodes_list.append(graph.num_nodes())
                num_edges_list.append(graph.num_edges())
            elif 'num_nodes' in sample:
                num_nodes_list.append(sample['num_nodes'])
            if 'num_edges' in sample:
                num_edges_list.append(sample['num_edges'])
        
        if num_nodes_list:
            stats['graph_statistics'] = {
                'avg_nodes': np.mean(num_nodes_list),
                'std_nodes': np.std(num_nodes_list),
                'min_nodes': np.min(num_nodes_list),
                'max_nodes': np.max(num_nodes_list),
                'total_nodes': sum(num_nodes_list)
            }
        
        if num_edges_list:
            stats['graph_statistics']['avg_edges'] = np.mean(num_edges_list)
            stats['graph_statistics']['std_edges'] = np.std(num_edges_list)
            stats['graph_statistics']['min_edges'] = np.min(num_edges_list)
            stats['graph_statistics']['max_edges'] = np.max(num_edges_list)
            stats['graph_statistics']['total_edges'] = sum(num_edges_list)
        
        return stats
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """
        验证数据样本是否有效
        
        Args:
            sample: 数据样本
            
        Returns:
            样本是否有效
        """
        try:
            # 基本字段检查
            required_fields = ['id']
            for field in required_fields:
                if field not in sample:
                    return False
            
            # 图数据检查
            if 'dgl_graph' in sample:
                graph = sample['dgl_graph']
                if not isinstance(graph, dgl.DGLGraph):
                    return False
                if graph.num_nodes() == 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_token_readable(self, token_id: int) -> str:
        """
        获取token到可读字符串的映射（可选实现）

        Returns:
            Dict[int, str]: 从token ID到可读字符串的映射
        """
        raise NotImplementedError(f"数据集 {self.dataset_name} 不支持token到可读字符串的映射")

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        计算类别权重（用于处理类别不平衡）

        这个方法应该在数据加载完成后调用，确保_train_data可用

        Returns:
            torch.Tensor: 类别权重向量，形状为[num_classes]
            如果不是分类任务或无法计算权重，返回None
        """
        if self._train_data is None:
            self.load_data()

        # 检查是否为分类任务
        task_type = self.get_dataset_task_type()
        if task_type not in ['classification', 'binary_classification']:
            return None

        # 检查是否有训练标签
        if not self._train_data:
            return None

        # 统计训练集类别分布
        train_labels = self._extract_labels(self._train_data)
        if not train_labels:
            return None

        # 计算类别权重
        num_classes = self.get_num_classes()
        class_counts = torch.zeros(num_classes, dtype=torch.float)

        for label in train_labels:
            if isinstance(label, (int, torch.Tensor)):
                label_idx = int(label) if isinstance(label, torch.Tensor) else label
                if 0 <= label_idx < num_classes:
                    class_counts[label_idx] += 1

        # 计算权重：total_samples / (num_classes * class_count)
        total_samples = len(train_labels)
        class_weights = torch.zeros(num_classes, dtype=torch.float)

        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (num_classes * class_counts[i])
            else:
                # 处理缺失类别的情况
                class_weights[i] = 1.0

        return class_weights
      
    # 注意：get_most_frequent_edge_type 必须由子类实现（见上方 abstractmethod 声明）
        
    def expand_tokens(self, token_lists: List[List[int]]) -> List[int]:
        """
        将多个token list展开为flat token sequence
        
        Args:
            token_lists: 多个token列表的列表
            
        Returns:
            List[int]: 展开的token序列
        """
        result = []
        for token_list in token_lists:
            result.extend(token_list)
        return result
