"""
Data Loading for Token ID Sequences
支持token ID序列输入的数据加载器
"""

import logging
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .vocab_manager import VocabManager
from .transforms import TokenTransform, Compose


class NoOpTransform(TokenTransform):
    """空操作Transform，用于确保transform pipeline总是存在"""
    def __init__(self):
        super().__init__(probability=1.0)

    def __call__(self, token_sequence: List[int]) -> List[int]:
        """直接返回原始序列，不做任何修改"""
        return token_sequence


logger = logging.getLogger(__name__)


def _candidate_len_from_policy(lengths: List[int], project_config) -> int:
    """根据配置策略返回候选长度（不含特殊token +2）。"""
    policy = project_config.bert.architecture.max_len_policy
    policy = str(policy).lower()
    if policy == 'sigma':
        # 支持 'sigma' 或 'sigma3'，k 可通过配置提供
        k = project_config.bert.architecture.max_len_sigma_k
        import math as _math
        import numpy as _np
        arr = _np.asarray(lengths, dtype=_np.int64)
        mean = float(arr.mean()) if arr.size > 0 else 0.0
        std = float(arr.std(ddof=0)) if arr.size > 0 else 0.0
        return int(_math.ceil(mean + k * std))
    # 默认：max
    return max(lengths) if lengths else 0


def compute_effective_max_length(token_sequences: List[List[int]], project_config, split_name: Optional[str] = None) -> int:
    """根据数据与配置计算有效的最大序列长度。

    规则：
      - config.bert.architecture.max_seq_length 作为上限（硬上限）
      - 数据侧长度优先：若 (max_len_from_data + 2) 不超过上限，则取 (max_len_from_data + 2)
      - 若数据存在极长序列导致 (max_len_from_data + 2) 超过上限，则取配置的上限
      - 严格检查位置嵌入：有效长度不得超过 config.bert.architecture.max_position_embeddings

    参数：
      - token_sequences: 序列列表
      - config: ProjectConfig 实例
      - split_name: 可选的分割名，仅用于日志

    返回：
      - int: 有效最大长度
    """
    assert token_sequences, "token_sequences 不能为空"

    lengths = [len(seq) for seq in token_sequences]
    data_candidate = _candidate_len_from_policy(lengths, project_config)
    data_max_plus_2 = int(data_candidate) + 2
    upper_bound = int(project_config.bert.architecture.max_seq_length)
    if project_config.encoder.type == 'bert':
      upper_bound = 768
    
    max_pos = int(project_config.bert.architecture.max_position_embeddings)

    effective = data_max_plus_2 if data_max_plus_2 <= upper_bound else upper_bound

    if effective > max_pos:
        name = f" ({split_name})" if split_name else ""
        raise ValueError(
            f"有效最大长度{name}为 {effective}，超过了 max_position_embeddings({max_pos})。"
            f"请增大 bert.architecture.max_position_embeddings 或降低 bert.architecture.max_seq_length。"
        )

    policy = project_config.bert.architecture.max_len_policy
    if split_name:
        logger.info(
            f"✅ 有效最大长度{f' [{split_name}]' if split_name else ''}: {effective}"
            f" (策略: {policy}, 数据侧: {data_max_plus_2}, 上限: {upper_bound}, 位置嵌入: {max_pos})"
        )
    else:
        logger.info(
            f"✅ 有效最大长度: {effective}"
            f" (策略: {policy}, 数据侧: {data_max_plus_2}, 上限: {upper_bound}, 位置嵌入: {max_pos})"
        )

    return effective

class MLMDataset(Dataset):
    """Token ID序列的MLM数据集"""

    def __init__(self, token_sequences: List[List[int]], vocab_manager: VocabManager,
                 transforms: TokenTransform, max_length: int = 512, mlm_probability: float = 0.15,
                 graph_ids: Optional[List[int]] = None, group_by_graph: bool = False,
                 variant_selection: str = "random"):
        self.token_sequences: List[List[int]] = token_sequences
        self.vocab_manager = vocab_manager
        self.max_length = min(max_length, max(len(seq) for seq in token_sequences)+2)
        self.mlm_probability: float = mlm_probability
        self.transforms = transforms
        self.graph_ids = graph_ids if graph_ids is not None else list(range(len(token_sequences)))
        self.group_by_graph = bool(group_by_graph)
        self.variant_selection = str(variant_selection).lower()

        # 图级采样支持：构建gid分组
        if self.group_by_graph:
            from collections import OrderedDict
            gid_to_indices = OrderedDict()
            for idx, gid in enumerate(self.graph_ids):
                gid_to_indices.setdefault(int(gid), []).append(idx)
            self._gid_to_indices = gid_to_indices
            self._unique_gids = list(gid_to_indices.keys())
            logger.info(f"🔧 图级采样启用: {len(self._unique_gids)} 个图，变体选择策略: {self.variant_selection}")

        # BPE Transform将在worker_init_fn中初始化，这里先设为None
        self._bpe_transform = None
        self._bpe_checked = False  # 标记是否已检查过BPE Transform

        # print(f"MLM数据集创建完成，共 {len(token_sequences)} 个序列")
    
    def _apply_bpe_if_enabled(self, token_sequence: List[int]) -> List[int]:
        """应用BPE编码（延迟初始化，失败时报错）"""
        if not self._bpe_checked:
            try:
                from src.data.bpe_transform import _g_bpe_transform
                self._bpe_transform = _g_bpe_transform
                if self._bpe_transform is None:
                    raise RuntimeError("BPE Transform未正确初始化（_g_bpe_transform为None）")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"BPE Transform初始化失败: {e}. 请检查worker_init_fn是否正确设置")
            self._bpe_checked = True
        
        return self._bpe_transform.encode(token_sequence)

    def __len__(self):
        if self.group_by_graph:
            return len(self._unique_gids)
        return len(self.token_sequences)
    
    def _create_mlm_mask(self, input_ids: torch.Tensor, 
                        attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建MLM掩码"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # 不对特殊token和padding进行mask
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        special_tokens_mask[input_ids == self.vocab_manager.cls_token_id] = True
        special_tokens_mask[input_ids == self.vocab_manager.sep_token_id] = True
        special_tokens_mask[input_ids == self.vocab_manager.pad_token_id] = True
        special_tokens_mask[attention_mask == 0] = True
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80%的时间：替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.vocab_manager.mask_token_id
        
        # 10%的时间：替换为随机token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_manager.vocab_size, labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        return input_ids, labels
    
    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        if self.group_by_graph:
            gid = self._unique_gids[idx]
            indices = self._gid_to_indices[gid]
            if self.variant_selection == "first":
                chosen_idx = indices[0]
            else:
                chosen_idx = random.choice(indices)
            token_sequence = self.token_sequences[chosen_idx]
        else:
            token_sequence = self.token_sequences[idx]

        # 应用数据增强变换
        token_sequence = self.transforms(token_sequence)

        # 应用BPE编码（动态检查，避免初始化时序问题）
        token_sequence = self._apply_bpe_if_enabled(token_sequence)

        # 使用词表管理器编码序列
        encoded: torch.Dict[str, torch.Tensor] = self.vocab_manager.encode_sequence(
            token_sequence, add_special_tokens=True, max_length=self.max_length
        )

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # 创建MLM掩码
        masked_input_ids, labels = self._create_mlm_mask(input_ids, attention_mask)

        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }






def create_mlm_dataloader(token_sequences: List[List[int]], project_config, vocab_manager: VocabManager,
                         batch_size: int = 8, max_length: int = 512,
                         mlm_probability: float = 0.15, shuffle: bool = True,
                         ) -> DataLoader:
    """创建MLM数据加载器"""
    
    # 获取有效的token列表，用于数据增强
    valid_tokens = vocab_manager.get_valid_tokens()
    transforms = create_transforms_from_config(project_config, valid_tokens, "mlm",logger=logger)

    
    dataset = MLMDataset(token_sequences, vocab_manager, transforms, max_length, mlm_probability)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)




class LabelNormalizer:
    """标签标准化器"""
    
    def __init__(self, method: str = 'standard'):
        """
        初始化标准化器
        
        Args:
            method: 标准化方法
                - 'standard': StandardScaler (Z-score标准化)
                - 'minmax': MinMaxScaler (最小-最大标准化)
                - 'robust': RobustScaler (鲁棒标准化)
        """
        self.method = method
        self.scaler = None
        self.is_fitted = False
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    def fit(self, labels: List[float]) -> 'LabelNormalizer':
        """
        拟合标准化器
        
        Args:
            labels: 原始标签列表（单目标：List[float]，多目标：List[List[float]]）
            
        Returns:
            self: 链式调用
        """
        labels_array = np.array(labels)
        
        # 检查是否为多目标回归
        if labels_array.ndim == 1:
            # 单目标回归：reshape为[N, 1]
            labels_array = labels_array.reshape(-1, 1)
        elif labels_array.ndim == 2:
            # 多目标回归：保持[N, num_targets]形状
            pass
        else:
            raise ValueError(f"标签数组维度不支持: {labels_array.ndim}")
        
        # 检查方差是否接近0
        if np.var(labels_array) < 1e-8:
            logger.warning("Label variance near zero! Adding epsilon to avoid division by zero")
            labels_array += np.random.normal(0, 1e-6, size=labels_array.shape)
        
        self.scaler.fit(labels_array)
        self.is_fitted = True
        
        logger.info(f"✅ 标签标准化器拟合完成: {self.method}")
        logger.info(f"   原始标签范围: [{labels_array.min():.6f}, {labels_array.max():.6f}],mean={labels_array.mean():.6f},std={labels_array.std():.6f}")
        logger.info(f"   标准化后范围: [{self.scaler.transform(labels_array).min():.6f}, {self.scaler.transform(labels_array).max():.6f}],mean={self.scaler.transform(labels_array).mean():.6f},std={self.scaler.transform(labels_array).std():.6f}")
        
        return self
    
    def transform(self, labels: List[float]) -> List[float]:
        """
        标准化标签
        
        Args:
            labels: 原始标签列表（单目标：List[float]，多目标：List[List[float]]）
            
        Returns:
            标准化后的标签列表（保持原有格式）
        """
        assert self.is_fitted, "标准化器尚未拟合，请先调用fit()方法"
        
        labels_array = np.array(labels)        
        # 处理不同维度
        if labels_array.ndim == 1:
            # 单目标回归：reshape为[N, 1]
            labels_array = labels_array.reshape(-1, 1)
            normalized_array = self.scaler.transform(labels_array)
            return normalized_array.flatten().tolist()
        elif labels_array.ndim == 2:
            # 多目标回归：直接transform
            normalized_array = self.scaler.transform(labels_array)
            return normalized_array.tolist()
        else:
            raise ValueError(f"标签数组维度不支持: {labels_array.ndim}")
    
    def inverse_transform(self, normalized_labels: List[float]) -> List[float]:
        """
        反标准化标签
        
        Args:
            normalized_labels: 标准化后的标签列表（单目标：List[float]，多目标：List[List[float]]）
            
        Returns:
            原始空间的标签列表（保持原有格式）
        """
        assert self.is_fitted, "标准化器尚未拟合，请先调用fit()方法"
        
        labels_array = np.array(normalized_labels)
        
        # 处理不同维度
        if labels_array.ndim == 1:
            # 单目标回归：reshape为[N, 1]
            labels_array = labels_array.reshape(-1, 1)
            original_array = self.scaler.inverse_transform(labels_array)
            return original_array.flatten().tolist()
        elif labels_array.ndim == 2:
            # 多目标回归：直接transform
            original_array = self.scaler.inverse_transform(labels_array)
            return original_array.tolist()
        else:
            raise ValueError(f"标签数组维度不支持: {labels_array.ndim}")
    
    def save(self, path: str):
        """保存标准化器"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"💾 标准化器已保存: {path}")
    
    def load(self, path: str):
        """加载标准化器"""
        import pickle
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        logger.info(f"📂 标准化器已加载: {path}")

class NormalizedRegressionDataset:
    """带标准化的回归数据集"""
    
    def __init__(self, token_sequences: List[List[int]], labels: List[float],
                 vocab_manager: VocabManager, transforms: TokenTransform, max_length: int = 512,
                 normalizer: LabelNormalizer = None, graph_ids: Optional[List[int]] = None,
                 group_by_graph: bool = False, variant_selection: str = "random"):
        """
        初始化标准化回归数据集
        
        Args:
            token_sequences: Token序列列表
            labels: 原始标签列表
            vocab_manager: 词表管理器
            max_length: 最大序列长度
            normalizer: 标签标准化器
            transforms: 数据增强变换
            graph_ids: 原始图ID列表，用于评估时聚合
        """
        self.token_sequences = token_sequences
        self.original_labels = labels
        self.graph_ids = graph_ids if graph_ids is not None else list(range(len(token_sequences)))
        self.vocab_manager = vocab_manager
        self.max_length = min(max_length, max(len(seq) for seq in token_sequences)+2)
        self.normalizer = normalizer
        self.transforms = transforms
        self.group_by_graph = bool(group_by_graph)
        self.variant_selection = str(variant_selection).lower()
        
        # 初始化时不进行标准化，避免数据泄露
        self.normalized_labels = None
        
        # BPE Transform将在worker_init_fn中初始化，这里先设为None
        self._bpe_transform = None
        self._bpe_checked = False  # 标记是否已检查过BPE Transform
        
        # 检查是否为多目标回归
        self.is_multi_target = isinstance(labels[0], (list, tuple, torch.Tensor)) if len(labels) > 0 else False

        # 若启用图级采样，构建 gid -> indices 的映射与图级标签
        if self.group_by_graph:
            from collections import OrderedDict
            gid_to_indices = OrderedDict()
            for idx, gid in enumerate(self.graph_ids):
                gid_to_indices.setdefault(int(gid), []).append(idx)
            self._gid_to_indices = gid_to_indices
            self._unique_gids = list(gid_to_indices.keys())
        
        logger.info(f"📊 回归数据集创建完成: {len(token_sequences)} 个序列")
        if self.group_by_graph:
            logger.info(f"   图级采样启用: {len(self._unique_gids)} 个图，选择策略: {self.variant_selection}")
        
        if self.is_multi_target:
            # 多目标回归：显示目标维度信息
            num_targets = len(labels[0]) if len(labels) > 0 else 1
            logger.info(f"   多目标回归，目标维度: {num_targets}")
            # 计算每个目标的范围（转换为numpy数组便于计算）
            import numpy as np
            labels_array = np.array(labels)
            min_vals = labels_array.min(axis=0)
            max_vals = labels_array.max(axis=0)
            logger.info(f"   各目标范围: min={min_vals[:3]}..., max={max_vals[:3]}...")
        else:
            # 单目标回归：原有逻辑
            logger.info(f"   原始标签范围: [{min(labels):.6f}, {max(labels):.6f}]")
        
        logger.info(f"   数据增强: {type(transforms).__name__}")
    
    def _apply_bpe_if_enabled(self, token_sequence: List[int]) -> List[int]:
        """应用BPE编码（延迟初始化，失败时报错）"""
        if not self._bpe_checked:
            try:
                from src.data.bpe_transform import _g_bpe_transform
                self._bpe_transform = _g_bpe_transform
                if self._bpe_transform is None:
                    raise RuntimeError("BPE Transform未正确初始化（_g_bpe_transform为None）")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"BPE Transform初始化失败: {e}. 请检查worker_init_fn是否正确设置")
            self._bpe_checked = True
        
        return self._bpe_transform.encode(token_sequence)

    def apply_normalization(self):
        """应用标准化（在normalizer已拟合后调用）"""
        if self.normalizer is not None and self.normalizer.is_fitted:
            self.normalized_labels = self.normalizer.transform(self.original_labels)
            # 处理多目标vs单目标的日志记录
            if self.is_multi_target:
                logger.info(f"   多目标标准化完成，目标维度: {len(self.normalized_labels[0])}")
            else:
                logger.info(f"   标准化后范围: [{min(self.normalized_labels):.6f}, {max(self.normalized_labels):.6f}]")
        elif self.normalizer is not None and not self.normalizer.is_fitted:
            raise ValueError("❌ 标准化器未拟合！请先调用normalizer.fit()")
        else:
            self.normalized_labels = self.original_labels
            logger.warning("⚠️ 未提供标准化器，使用原始标签")
    
    def __len__(self):
        if self.group_by_graph:
            return len(self._unique_gids)
        return len(self.token_sequences)
    
    def __getitem__(self, idx):
        if self.group_by_graph:
            gid = self._unique_gids[idx]
            indices = self._gid_to_indices[gid]
            if self.variant_selection == "first":
                chosen_idx = indices[0]
            else:
                chosen_idx = random.choice(indices)
            token_sequence = self.token_sequences[chosen_idx]
            # 应用数据增强与BPE
            token_sequence = self.transforms(token_sequence)
            token_sequence = self._apply_bpe_if_enabled(token_sequence)
            # 确保已应用标准化
            if self.normalized_labels is None:
                self.apply_normalization()
            normalized_label = self.normalized_labels[chosen_idx]
            encoded = self.vocab_manager.encode_sequence(
                token_sequence, add_special_tokens=True, max_length=self.max_length
            )
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(normalized_label, dtype=torch.float),
                'original_label': torch.tensor(float(self.original_labels[chosen_idx]), dtype=torch.float),
                'graph_id': torch.tensor(gid, dtype=torch.long)
            }
        else:
            token_sequence = self.token_sequences[idx]
            
            # 应用数据增强变换
            token_sequence = self.transforms(token_sequence)
            
            # 应用BPE编码（动态检查，避免初始化时序问题）
            token_sequence = self._apply_bpe_if_enabled(token_sequence)
            
            # 确保已应用标准化
            if self.normalized_labels is None:
                self.apply_normalization()
            
            normalized_label = self.normalized_labels[idx]
            
            # 使用词表管理器编码序列
            encoded = self.vocab_manager.encode_sequence(
                token_sequence, add_special_tokens=True, max_length=self.max_length
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.tensor(normalized_label, dtype=torch.float),
                'original_label': torch.tensor(float(self.original_labels[idx]), dtype=torch.float),
                'graph_id': torch.tensor(self.graph_ids[idx], dtype=torch.long)
            }



def create_transforms_from_config(project_config, valid_tokens, task_type: str = "mlm", vocab_manager=None,logger=None) -> TokenTransform:
    """根据配置创建统一的transform pipeline，总是返回有效的transform"""
    
    # 获取指定任务的方法列表和增强配置
    if task_type == "mlm":
        methods = project_config.bert.pretraining.mlm_augmentation_methods
        aug_config = project_config.bert.pretraining.augmentation_config
    else:
        methods = project_config.bert.finetuning.regression_augmentation_methods
        aug_config = project_config.bert.finetuning.augmentation_config
    
    # 根据指定的方法创建transforms
    transforms = []
    
    # 添加数据增强transforms
    if methods:
        for method in methods:
            if method == "random_deletion":
                from .transforms import RandomDeletion
                transforms.append(RandomDeletion(
                    deletion_ratio=aug_config.random_deletion_ratio,
                    probability=aug_config.random_deletion_probability
                ))
            elif method == "random_swap":
                from .transforms import RandomSwap
                transforms.append(RandomSwap(
                    swap_ratio=aug_config.random_swap_ratio,
                    probability=aug_config.random_swap_probability,
                    window_size=aug_config.swap_window_size
                ))
            elif method == "random_truncation":
                from .transforms import RandomTruncation
                transforms.append(RandomTruncation(
                    min_ratio=aug_config.random_truncation_min_ratio,
                    probability=aug_config.random_truncation_probability
                ))
            elif method == "sequence_masking":
                from .transforms import SequenceMasking
                sequence_masking = SequenceMasking(
                    mask_ratio=aug_config.sequence_masking_ratio,
                    probability=aug_config.sequence_masking_probability
                )
                # 设置mask token ID（从vocab_manager获取）
                if vocab_manager is not None and hasattr(vocab_manager, 'mask_token_id'):
                    sequence_masking.set_mask_token_id(vocab_manager.mask_token_id)
                transforms.append(sequence_masking)
            else:
                print(f"警告：未知的数据增强方法 '{method}'，跳过")
    
    # 总是返回有效的transform pipeline
    if transforms:
        logger.info(f"🔄 创建Transform pipeline，包含数据增强: {[type(t).__name__ for t in transforms]}")
        return Compose(transforms)
    else:
        logger.info("🔄 创建Transform pipeline，使用NoOp transform")
        return NoOpTransform() 

class ClassificationDataset(Dataset):
    """分类任务数据集"""
    
    def __init__(self, token_sequences: List[List[int]], labels: List[int],
                 vocab_manager: VocabManager, transforms: TokenTransform, max_length: int = 512,
                 graph_ids: Optional[List[int]] = None,
                 group_by_graph: bool = False, variant_selection: str = "random"):
        """
        初始化分类数据集
        
        Args:
            token_sequences: Token序列列表
            labels: 分类标签列表 (整数)
            vocab_manager: 词表管理器
            max_length: 最大序列长度
            transforms: 数据增强变换
            graph_ids: 原始图ID列表
        """
        self.token_sequences = token_sequences
        self.labels = labels
        self.graph_ids = graph_ids if graph_ids is not None else list(range(len(token_sequences)))
        self.vocab_manager = vocab_manager
        self.max_length = min(max_length, max(len(seq) for seq in token_sequences)+2)
        self.transforms = transforms
        self.group_by_graph = bool(group_by_graph)
        self.variant_selection = str(variant_selection).lower()
        
        # 验证数据
        assert len(token_sequences) == len(labels), "序列数量和标签数量不匹配"
        
        # 检查标签类型：多标签 vs 单标签
        self.is_multi_label = isinstance(labels[0], (list, tuple, torch.Tensor)) if len(labels) > 0 else False

        # 图级采样支持：构建gid分组与图级标签
        if self.group_by_graph:
            from collections import OrderedDict
            gid_to_indices = OrderedDict()
            for idx, gid in enumerate(self.graph_ids):
                gid_to_indices.setdefault(int(gid), []).append(idx)
            self._gid_to_indices = gid_to_indices
            self._unique_gids = list(gid_to_indices.keys())
            # 图级标签
            grouped_labels = []
            for gid in self._unique_gids:
                first_idx = gid_to_indices[gid][0]
                grouped_labels.append(self.labels[first_idx])
            self._grouped_labels = grouped_labels
        
        if self.is_multi_label:
            # 多标签分类：标签是向量
            base_label = labels[0] if len(labels) > 0 else []
            self.num_classes = len(base_label) if base_label is not None else 1
            self.class_distribution = "多标签分类"
            print(f"多标签分类数据集创建完成，共 {len(token_sequences)} 个样本")
            print(f"标签维度: {self.num_classes}")
        else:
            # 单标签分类：统计类别信息
            if self.group_by_graph:
                unique_labels = sorted(set(labels))
                self.num_classes = len(unique_labels)
                self.class_distribution = {label: labels.count(label) for label in unique_labels}
                print(f"分类数据集创建完成，共 {len(self._unique_gids)} 个图 (图级采样，训练期每图随机选1变体)")
            else:
                unique_labels = sorted(set(labels))
                self.num_classes = len(unique_labels)
                self.class_distribution = {label: labels.count(label) for label in unique_labels}
                print(f"分类数据集创建完成，共 {len(token_sequences)} 个样本")
            print(f"类别数量: {self.num_classes}, 类别分布: {self.class_distribution}")
        
        print(f"序列长度固定为: {self.max_length}") 
        
        # BPE Transform将在worker_init_fn中初始化，这里先设为None
        self._bpe_transform = None
        self._bpe_checked = False  # 标记是否已检查过BPE Transform
    
    def _apply_bpe_if_enabled(self, token_sequence: List[int]) -> List[int]:
        """应用BPE编码（延迟初始化，失败时报错）"""
        if not self._bpe_checked:
            try:
                from src.data.bpe_transform import _g_bpe_transform
                self._bpe_transform = _g_bpe_transform
                if self._bpe_transform is None:
                    raise RuntimeError("BPE Transform未正确初始化（_g_bpe_transform为None）")
            except (ImportError, AttributeError) as e:
                raise RuntimeError(f"BPE Transform初始化失败: {e}. 请检查worker_init_fn是否正确设置")
            self._bpe_checked = True
        
        return self._bpe_transform.encode(token_sequence)

    def __len__(self):
        if self.group_by_graph:
            return len(self._unique_gids)
        return len(self.token_sequences)
    
    def __getitem__(self, idx):
        if self.group_by_graph:
            gid = self._unique_gids[idx]
            indices = self._gid_to_indices[gid]
            if self.variant_selection == "first":
                chosen_idx = indices[0]
            else:
                chosen_idx = random.choice(indices)
            token_sequence = self.token_sequences[chosen_idx]
            label = self._grouped_labels[idx]
            token_sequence = self.transforms(token_sequence)
            token_sequence = self._apply_bpe_if_enabled(token_sequence)
            encoded = self.vocab_manager.encode_sequence(
                token_sequence, add_special_tokens=True, max_length=self.max_length
            )
            if self.is_multi_label:
                label_tensor = torch.tensor(label, dtype=torch.float)
            else:
                label_tensor = torch.tensor(label, dtype=torch.long)
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': label_tensor,
                'graph_id': torch.tensor(gid, dtype=torch.long)
            }
        else:
            token_sequence = self.token_sequences[idx]
            label = self.labels[idx]
            
            # 应用数据增强变换
            token_sequence = self.transforms(token_sequence)
            
            # 应用BPE编码（动态检查，避免初始化时序问题）
            token_sequence = self._apply_bpe_if_enabled(token_sequence)
            
            # 使用词表管理器编码序列
            encoded = self.vocab_manager.encode_sequence(
                token_sequence, add_special_tokens=True, max_length=self.max_length
            )
            
            # 根据标签类型选择合适的dtype
            if self.is_multi_label:
                # 多标签分类：使用float类型
                label_tensor = torch.tensor(label, dtype=torch.float)
            else:
                # 单标签分类：使用long类型
                label_tensor = torch.tensor(label, dtype=torch.long)
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': label_tensor,
                'graph_id': torch.tensor(self.graph_ids[idx], dtype=torch.long)
            }
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        if self.is_multi_label:
            # 多标签分类：返回均匀权重
            return torch.ones(self.num_classes)
        else:
            # 单标签分类：计算基于频率的权重
            if self.group_by_graph:
                total_samples = len(self._grouped_labels)
            else:
                total_samples = len(self.labels)
            class_weights = torch.zeros(self.num_classes)
            for label, count in self.class_distribution.items():
                class_weights[label] = total_samples / (self.num_classes * count)
            return class_weights

def create_classification_dataloader(token_sequences: List[List[int]], 
                                   labels: List[int],
                                   vocab_manager: VocabManager, 
                                   transforms: TokenTransform,
                                   batch_size: int = 8,
                                   max_length: int = 512, 
                                   shuffle: bool = True) -> torch.utils.data.DataLoader:
    """创建分类数据加载器"""
    dataset = ClassificationDataset(
        token_sequences=token_sequences,
        labels=labels,
        vocab_manager=vocab_manager,
        transforms=transforms,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    ) 