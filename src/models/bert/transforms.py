"""
Data augmentation transforms for token sequences.
Token序列的数据增强变换

Composable augmentation ops similar to CV transforms.
类似CV中的transforms，提供可组合的数据增强操作。

Supported sequence-level augmentations / 支持的序列级增强:
- RandomDeletion: delete a fraction of tokens / 删除指定比例的token
- RandomSwap: swap tokens within a local window / 在局部窗口内交换token
- RandomTruncation: randomly crop a subsequence / 随机截取序列片段
- SequenceMasking: replace tokens with [MASK] (fine-tuning) / 用[MASK]替换部分token（微调阶段）

Config: set in config/default_config.yml under augmentation_config
配置方式：在config/default_config.yml的augmentation_config中设置
"""

import random
from typing import List
from abc import ABC, abstractmethod


class TokenTransform(ABC):
    """Base class for token sequence transforms.
    Token序列变换的基类。"""
    
    def __init__(self, probability: float = 0.3):
        self.probability = probability
    
    @abstractmethod
    def __call__(self, sequence: List[int]) -> List[int]:
        """Apply transform to a sequence.
        应用变换到序列。"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(probability={self.probability})"


class RandomDeletion(TokenTransform):
    """Randomly delete tokens.
    随机删除token。"""
    
    def __init__(self, deletion_ratio: float = 0.1, probability: float = 0.1):
        super().__init__(probability)
        self.deletion_ratio = deletion_ratio
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
        
        # Number of tokens to delete (ratio-based) / 计算要删除的token数量（基于比例）
        num_to_delete = max(0, int(len(sequence) * self.deletion_ratio))
        if num_to_delete >= len(sequence):
            num_to_delete = len(sequence) - 1  # keep at least one token / 至少保留一个token
        
        if num_to_delete == 0:
            return sequence
            
        # Randomly select indices to delete / 随机选择要删除的索引
        delete_indices = set(random.sample(range(len(sequence)), num_to_delete))
        
        # Build the retained sequence / 构建保留的序列
        return [sequence[i] for i in range(len(sequence)) if i not in delete_indices]


class RandomInsertion(TokenTransform):
    """Randomly insert tokens.
    随机插入token。"""
    
    def __init__(self, valid_tokens, insertion_ratio: float = 0.05, probability: float = 0.1):
        super().__init__(probability)
        self.insertion_ratio = insertion_ratio
        # Use actual token list instead of vocab_size range / 使用实际存在的token列表
        self.valid_tokens = valid_tokens 
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability:
            return sequence
        
        # Number of tokens to insert (ratio-based) / 计算要插入的token数量
        num_to_insert = max(0, int(len(sequence) * self.insertion_ratio))
        if num_to_insert == 0:
            return sequence
            
        # Randomly select insertion positions (incl. start and end) / 随机选择插入位置
        insert_positions = sorted(random.choices(range(len(sequence) + 1), k=num_to_insert), reverse=True)
        
        augmented = sequence.copy()
        # Insert back-to-front to avoid index shift / 从后往前插入避免索引变化
        for pos in insert_positions:
            random_token = random.choice(self.valid_tokens)
            augmented.insert(pos, random_token)
        
        return augmented


class RandomReplacement(TokenTransform):
    """Randomly replace tokens.
    随机替换token。"""
    
    def __init__(self, valid_tokens, replacement_ratio: float = 0.05, probability: float = 0.1):
        super().__init__(probability)
        self.replacement_ratio = replacement_ratio
        # Use actual token list instead of vocab_size range / 使用实际存在的token列表
        self.valid_tokens = valid_tokens 
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability:
            return sequence
        
        # Number of tokens to replace (ratio-based) / 计算要替换的token数量
        num_to_replace = max(0, int(len(sequence) * self.replacement_ratio))
        if num_to_replace == 0:
            return sequence
            
        # Randomly select indices to replace / 随机选择要替换的索引
        replace_indices = random.sample(range(len(sequence)), min(num_to_replace, len(sequence)))
        
        augmented = sequence.copy()
        for i in replace_indices:
            augmented[i] = random.choice(self.valid_tokens)
        
        return augmented


class RandomSwap(TokenTransform):
    """Randomly swap tokens within a local window.
    在局部窗口内随机交换token。"""
    
    def __init__(self, swap_ratio: float = 0.05, probability: float = 0.1, window_size: int = 3):
        super().__init__(probability)
        self.swap_ratio = swap_ratio
        self.window_size = window_size
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
        
        # Number of swap operations (ratio-based) / 要执行交换的数量
        num_swaps = max(0, int(len(sequence) * self.swap_ratio))
        
        if num_swaps == 0:
            return sequence
        
        augmented = sequence.copy()
        
        # Perform windowed swaps / 对选定数量的token执行窗口内交换
        for _ in range(num_swaps):
            # Pick a random center position / 随机选择窗口中心
            center = random.randint(0, len(augmented) - 1)
            
            # Determine window bounds / 确定窗口范围
            window_start = max(0, center - self.window_size // 2)
            window_end = min(len(augmented), center + self.window_size // 2 + 1)
            
            # Swap two positions within the window if possible / 窗口内至少有2个token时交换
            if window_end - window_start >= 2:
                # Pick two distinct positions inside the window / 窗口内随机选两个位置交换
                pos1, pos2 = random.sample(range(window_start, window_end), 2)
                augmented[pos1], augmented[pos2] = augmented[pos2], augmented[pos1]
        
        return augmented


class RandomTruncation(TokenTransform):
    """Randomly crop a contiguous subsequence.
    随机截取序列的一部分。"""
    
    def __init__(self, min_ratio: float = 0.7, probability: float = 0.1):
        super().__init__(probability)
        self.min_ratio = min_ratio
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
        
        min_len = max(1, int(len(sequence) * self.min_ratio))
        if min_len >= len(sequence):
            return sequence
        
        start = random.randint(0, len(sequence) - min_len)
        end = random.randint(start + min_len, len(sequence))
        return sequence[start:end]


class SequenceMasking(TokenTransform):
    """Low-probability sequence masking (for fine-tuning).
    低概率序列掩码（适用于微调阶段）。"""
    
    def __init__(self, mask_ratio: float = 0.05, probability: float = 0.3, mask_token_id: int = -2):
        super().__init__(probability)
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id
    
    def set_mask_token_id(self, mask_token_id: int):
        """Set mask token ID (provided by external vocab_manager).
        设置掩码token ID（由外部vocab_manager提供）。"""
        self.mask_token_id = mask_token_id
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
            
        if self.mask_token_id is None:
            # Skip masking if mask_token_id is not set / 如果没有设置mask_token_id，跳过
            return sequence
        
        # Number of tokens to mask (at least 1, at most len-1) / 要掩码的token数量
        num_to_mask = max(1, min(int(len(sequence) * self.mask_ratio), len(sequence) - 1))
        
        # Randomly select positions to mask / 随机选择掩码位置
        mask_positions = random.sample(range(len(sequence)), num_to_mask)
        
        augmented = sequence.copy()
        for pos in mask_positions:
            augmented[pos] = self.mask_token_id
        
        return augmented


class Compose:
    """Compose multiple transforms.
    组合多个变换。"""
    
    def __init__(self, transforms: List[TokenTransform]):
        self.transforms = transforms
    
    def __call__(self, sequence: List[int]) -> List[int]:
        for transform in self.transforms:
            sequence = transform(sequence)
        return sequence
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


# Predefined transform presets / 预定义的变换组合
def get_default_transforms() -> Compose:
    """Get default transform composition.
    获取默认的变换组合。"""
    return Compose([
        RandomDeletion(deletion_ratio=0.05, probability=0.3),
        RandomSwap(swap_ratio=0.05, probability=0.3, window_size=3),
    ])


def get_aggressive_transforms() -> Compose:
    """Get aggressive transform composition.
    获取更激进的变换组合。"""
    return Compose([
        RandomDeletion(deletion_ratio=0.1, probability=0.5),
        RandomSwap(swap_ratio=0.1, probability=0.5, window_size=3),
        RandomTruncation(min_ratio=0.6, probability=0.3),
    ])


def get_conservative_transforms() -> Compose:
    """Get conservative transform composition.
    获取保守的变换组合。"""
    return Compose([
        RandomDeletion(deletion_ratio=0.02, probability=0.2),
        RandomSwap(swap_ratio=0.02, probability=0.2, window_size=3),
    ]) 