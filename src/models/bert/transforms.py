"""
Token序列的数据增强变换
类似CV中的transforms，提供可组合的数据增强操作
"""

import random
import torch
from typing import List, Union, Optional, Callable
from abc import ABC, abstractmethod


class TokenTransform(ABC):
    """Token序列变换的基类"""
    
    def __init__(self, probability: float = 0.3):
        self.probability = probability
    
    @abstractmethod
    def __call__(self, sequence: List[int]) -> List[int]:
        """应用变换到序列"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(probability={self.probability})"


class RandomDeletion(TokenTransform):
    """随机删除token"""
    
    def __init__(self, deletion_prob: float = 0.1, probability: float = 0.3):
        super().__init__(probability)
        self.deletion_prob = deletion_prob
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
        
        # 保留的token索引
        keep_indices = [i for i in range(len(sequence)) 
                       if random.random() > self.deletion_prob]
        
        if not keep_indices:  # 如果所有token都被删除，至少保留一个
            keep_indices = [random.randint(0, len(sequence) - 1)]
        
        return [sequence[i] for i in keep_indices]


class RandomInsertion(TokenTransform):
    """随机插入token"""
    
    def __init__(self, valid_tokens,insertion_prob: float = 0.1, probability: float = 0.3, ):
        super().__init__(probability)
        self.insertion_prob = insertion_prob
        # 使用实际存在的token列表，而不是vocab_size范围
        self.valid_tokens = valid_tokens 
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability:
            return sequence
        
        augmented = sequence.copy()
        # 从后往前插入，避免索引变化问题
        for i in range(len(sequence) - 1, -1, -1):
            if random.random() < self.insertion_prob:
                random_token = random.choice(self.valid_tokens)
                augmented.insert(i, random_token)
        
        return augmented


class RandomReplacement(TokenTransform):
    """随机替换token"""
    
    def __init__(self, valid_tokens, replacement_prob: float = 0.1, probability: float = 0.3):
        super().__init__(probability)
        self.replacement_prob = replacement_prob
        # 使用实际存在的token列表，而不是vocab_size范围
        self.valid_tokens = valid_tokens 
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability:
            return sequence
        
        augmented = sequence.copy()
        for i in range(len(augmented)):
            if random.random() < self.replacement_prob:
                augmented[i] = random.choice(self.valid_tokens)
        
        return augmented


class RandomSwap(TokenTransform):
    """随机交换相邻token"""
    
    def __init__(self, swap_prob: float = 0.1, probability: float = 0.3):
        super().__init__(probability)
        self.swap_prob = swap_prob
    
    def __call__(self, sequence: List[int]) -> List[int]:
        if random.random() > self.probability or len(sequence) <= 1:
            return sequence
        
        augmented = sequence.copy()
        for i in range(len(augmented) - 1):
            if random.random() < self.swap_prob:
                augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]
        
        return augmented


class RandomTruncation(TokenTransform):
    """随机截取序列的一部分"""
    
    def __init__(self, min_ratio: float = 0.7, probability: float = 0.3):
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


class Compose:
    """组合多个变换"""
    
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


# 预定义的变换组合
def get_default_transforms(vocab_size: int, valid_tokens: Optional[List[int]] = None) -> Compose:
    """获取默认的变换组合"""
    return Compose([
        RandomDeletion(deletion_prob=0.1, probability=0.3),
        RandomInsertion(vocab_size, insertion_prob=0.1, probability=0.3, valid_tokens=valid_tokens),
        RandomReplacement(vocab_size, replacement_prob=0.1, probability=0.3, valid_tokens=valid_tokens),
        RandomSwap(swap_prob=0.1, probability=0.3),
    ])


def get_aggressive_transforms(vocab_size: int, valid_tokens: Optional[List[int]] = None) -> Compose:
    """获取更激进的变换组合"""
    return Compose([
        RandomDeletion(deletion_prob=0.15, probability=0.5),
        RandomInsertion(vocab_size, insertion_prob=0.15, probability=0.5, valid_tokens=valid_tokens),
        RandomReplacement(vocab_size, replacement_prob=0.15, probability=0.5, valid_tokens=valid_tokens),
        RandomSwap(swap_prob=0.15, probability=0.5),
        RandomTruncation(min_ratio=0.6, probability=0.3),
    ])


def get_conservative_transforms(vocab_size: int, valid_tokens: Optional[List[int]] = None) -> Compose:
    """获取保守的变换组合"""
    return Compose([
        RandomDeletion(deletion_prob=0.05, probability=0.2),
        RandomInsertion(vocab_size, insertion_prob=0.05, probability=0.2, valid_tokens=valid_tokens),
        RandomReplacement(vocab_size, replacement_prob=0.05, probability=0.2, valid_tokens=valid_tokens),
        RandomSwap(swap_prob=0.05, probability=0.2),
    ]) 