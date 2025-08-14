"""
Vocabulary Manager for Custom BERT
自定义BERT的词表管理器
"""
# pyright: reportUnusedImport=false
# pyright: reportUnusedVariable=false

import json
import pickle
from typing import List, Dict, Set, Optional, Union # type: ignore
from collections import Counter
import torch
import warnings
import numpy as np
from config import ProjectConfig

class VocabManager:
    """词表管理器，从token ID序列构建和管理词表"""
    
    def __init__(self, config: ProjectConfig):
        """
        Args:
            config: 项目配置，用于获取标准化的特殊token
        """
        # 使用config中的标准化特殊token
        self.special_tokens = [
            config.pad_token,
            config.unk_token,
            config.mask_token,
            config.cls_token,
            config.sep_token,
            config.node_start_token,
            config.node_end_token,
            config.component_sep_token
        ]
        
        # 使用config中的标准化ID
        self.pad_token_id = config.pad_token_id
        self.unk_token_id = config.unk_token_id
        self.cls_token_id = config.cls_token_id
        self.sep_token_id = config.sep_token_id
        self.mask_token_id = config.mask_token_id
        self.node_start_token_id = config.node_start_token_id
        self.node_end_token_id = config.node_end_token_id
        self.component_sep_token_id = config.component_sep_token_id
        
        # token字符串属性
        self.pad_token = config.pad_token
        self.unk_token = config.unk_token
        self.cls_token = config.cls_token
        self.sep_token = config.sep_token
        self.mask_token = config.mask_token
        self.node_start_token = config.node_start_token
        self.node_end_token = config.node_end_token
        self.component_sep_token = config.component_sep_token
        # 词表相关
        self.token_to_id: Dict[int, int] = {}  # 原始token_id -> 新的vocab_id
        self.id_to_token: Dict[int, int] = {}  # 新的vocab_id -> 原始token_id
        self.vocab_size = 0
        
        # 预留特殊token位置
        self._reserve_special_tokens()
        
        # 统计信息
        self.token_counts: Counter = Counter()
        self._built = False
    
    def _reserve_special_tokens(self):
        """为特殊token预留位置"""
        # 使用标准化的特殊token ID映射
        special_id_map = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.cls_token: self.cls_token_id,
            self.sep_token: self.sep_token_id,
            self.mask_token: self.mask_token_id,
            self.node_start_token: self.node_start_token_id,
            self.node_end_token: self.node_end_token_id,
            self.component_sep_token: self.component_sep_token_id
        }
        
        for token in self.special_tokens:
            if token in special_id_map:
                special_id = -special_id_map[token]  # 用负数表示特殊token
                self.token_to_id[special_id] = special_id_map[token]
                self.id_to_token[special_id_map[token]] = special_id
        
        self.vocab_size = len(self.special_tokens)
    
    def add_token_sequences(self, token_sequences: List[List[int]]):
        """添加token序列到词表统计中
        
        Args:
            token_sequences: token ID序列列表 [[1,50,8909,...],[...],...]
        """
        for sequence in token_sequences:
            for token_id in sequence:
                # 统一转换为 python int；接受 numpy 整数类型
                if isinstance(token_id, (int, np.integer)) and int(token_id) >= 0:
                    self.token_counts[int(token_id)] += 1
    
    def build_vocab(self, min_freq: int = 1, max_vocab_size: Optional[int] = None):
        """构建最终词表
        
        Args:
            min_freq: 最小词频，低于此频率的token会被标记为UNK
            max_vocab_size: 最大词表大小（包含特殊token）
        """
        print(f"开始构建词表，统计到 {len(self.token_counts)} 个不同的token...")
        
        # 统计词频分布
        freq_distribution = {}
        for count in self.token_counts.values():
            freq_distribution[count] = freq_distribution.get(count, 0) + 1
        
        print(f"📊 词频分布统计:")
        print(f"   总token类型数: {len(self.token_counts)}")
        print(f"   最高频率: {max(self.token_counts.values()) if self.token_counts else 0}")
        print(f"   最低频率: {min(self.token_counts.values()) if self.token_counts else 0}")
        
        # 过滤低频token
        filtered_tokens = {token: count for token, count in self.token_counts.items() 
                          if count >= min_freq}
        
        # 详细的过滤统计
        filtered_out_count = len(self.token_counts) - len(filtered_tokens)
        print(f"过滤低频token后剩余 {len(filtered_tokens)} 个token (min_freq={min_freq})")
        if filtered_out_count > 0:
            print(f"⚠️  过滤掉 {filtered_out_count} 个低频token")
            # 显示被过滤掉的token示例
            filtered_out_tokens = [token for token, count in self.token_counts.items() if count < min_freq]
            if len(filtered_out_tokens) <= 20:
                print(f"   被过滤的tokens: {sorted(filtered_out_tokens)}")
            else:
                print(f"   被过滤的tokens示例(前20个): {sorted(filtered_out_tokens)[:20]}")
        
        # 按频率排序
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
        
        # 限制词表大小
        if max_vocab_size is not None:
            available_size = max_vocab_size - len(self.special_tokens)
            if len(sorted_tokens) > available_size:
                sorted_tokens = sorted_tokens[:available_size]
                print(f"限制词表大小到 {max_vocab_size}，保留高频token {len(sorted_tokens)} 个")
        
        # 构建映射关系
        current_id = len(self.special_tokens)  # 从特殊token之后开始
        
        for original_token_id, count in sorted_tokens:
            if original_token_id not in self.token_to_id:  # 避免重复添加
                self.token_to_id[original_token_id] = current_id
                self.id_to_token[current_id] = original_token_id
                current_id += 1
        
        self.vocab_size = current_id
        self._built = True
        
        print(f"词表构建完成！最终词表大小: {self.vocab_size}")
        print(f"特殊token: {len(self.special_tokens)}, 普通token: {self.vocab_size - len(self.special_tokens)}")
        
        # 添加词表覆盖率统计
        total_tokens_in_data = sum(self.token_counts.values())
        covered_tokens = sum(count for token, count in self.token_counts.items() 
                           if token in self.token_to_id)
        coverage_rate = covered_tokens / total_tokens_in_data if total_tokens_in_data > 0 else 0
        print(f"📈 词表覆盖率: {coverage_rate:.4f} ({covered_tokens}/{total_tokens_in_data})")
    
    def convert_token_to_id(self, token_id: int) -> int:
        """将原始token ID转换为词表中的ID"""
        if not self._built:
            raise ValueError("词表尚未构建，请先调用 build_vocab()")
        
        if token_id in self.token_to_id:
            return self.token_to_id[token_id]
        else:
            # 记录未知token用于调试
            if not hasattr(self, '_unknown_tokens_seen'):
                self._unknown_tokens_seen = set()
            self._unknown_tokens_seen.add(token_id)
            
            # 只在第一次遇到时发出警告，避免日志过多
            if len(self._unknown_tokens_seen) <= 10:  # 只显示前10个未知token
                warnings.warn(f"遇到未知token: {token_id} (词表中没有此token，将映射为UNK)")
            elif len(self._unknown_tokens_seen) == 11:
                warnings.warn(f"已遇到超过10个未知token，后续不再显示警告。总未知token数: {len(self._unknown_tokens_seen)}")
            
            return self.unk_token_id  # 未知token
    
    def convert_id_to_token(self, vocab_id: int) -> int:
        """将词表ID转换为原始token ID"""
        if not self._built:
            raise ValueError("词表尚未构建，请先调用 build_vocab()")
        
        if vocab_id in self.id_to_token:
            return self.id_to_token[vocab_id]
        else:
            warnings.warn(f"未知token: {vocab_id}")
            return self.id_to_token[self.unk_token_id]  # 默认返回UNK对应的原始ID
    
    def convert_tokens_to_ids(self, token_ids: List[int]) -> List[int]:
        """批量转换token ID"""
        return [self.convert_token_to_id(tid) for tid in token_ids]
    
    def convert_ids_to_tokens(self, vocab_ids: List[int]) -> List[int]:
        """批量转换vocab ID到原始token ID"""
        return [self.convert_id_to_token(vid) for vid in vocab_ids]
    
    def encode_sequence(self, token_sequence: List[int], 
                       add_special_tokens: bool = True,
                       max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """编码单个序列
        
        Args:
            token_sequence: 原始token ID序列
            add_special_tokens: 是否添加[CLS]和[SEP]
            max_length: 最大长度，超过会截断，不足会填充
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        if not self._built:
            raise ValueError("词表尚未构建，请先调用 build_vocab()")
        
        # 转换为词表ID
        sequence = self.convert_tokens_to_ids(token_sequence)
        
        # 添加特殊token
        if add_special_tokens:
            sequence = [self.cls_token_id] + sequence + [self.sep_token_id]
        
        # 截断
        if max_length is not None and len(sequence) > max_length:
            sequence = sequence[:max_length-1] + [self.sep_token_id]
        
        # 创建attention mask
        attention_mask = [1] * len(sequence)
        
        # 填充
        if max_length is not None:
            while len(sequence) < max_length:
                sequence.append(self.pad_token_id)
                attention_mask.append(0)
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def encode_batch(self, token_sequences: List[List[int]], 
                    add_special_tokens: bool = True,
                    max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """批量编码序列"""
        encoded_batch = [self.encode_sequence(seq, add_special_tokens, max_length) 
                        for seq in token_sequences]
        
        return {
            'input_ids': torch.stack([item['input_ids'] for item in encoded_batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in encoded_batch])
        }
    
    def get_vocab_info(self) -> Dict:
        """获取词表信息"""
        return {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'special_token_ids': {
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'cls_token_id': self.cls_token_id,
                'sep_token_id': self.sep_token_id,
                'mask_token_id': self.mask_token_id,
                'node_start_token_id': self.node_start_token_id,
                'node_end_token_id': self.node_end_token_id,
                'component_sep_token_id': self.component_sep_token_id
            },
            'total_tokens_seen': sum(self.token_counts.values()),
            'unique_tokens_seen': len(self.token_counts),
            'built': self._built
        }
    
    def get_valid_tokens(self) -> List[int]:
        """获取有效的token ID列表（排除特殊token）"""
        if not self._built:
            raise ValueError("词表尚未构建，请先调用 build_vocab()")
        
        # 返回所有非负整数的token ID（排除特殊token的负数ID）
        valid_tokens = []
        for original_token_id in self.token_to_id.keys():
            if original_token_id >= 0:  # 只包含非负整数token
                valid_tokens.append(original_token_id)
        
        return valid_tokens
    
    def save_vocab(self, save_path: str):
        """保存词表到文件"""
        if not self._built:
            raise ValueError("词表尚未构建，无法保存")
        
        vocab_data = {
            'special_tokens': self.special_tokens,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'token_counts': dict(self.token_counts),
            'special_token_ids': {
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'cls_token_id': self.cls_token_id,
                'sep_token_id': self.sep_token_id,
                'mask_token_id': self.mask_token_id,
                'node_start_token_id': self.node_start_token_id,
                'node_end_token_id': self.node_end_token_id,
                'component_sep_token_id': self.component_sep_token_id
            }
        }
        
        if save_path.endswith('.json'):
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(vocab_data, f)
        
        print(f"词表已保存到: {save_path}")
    
    @classmethod
    def load_vocab(cls, load_path: str, config: ProjectConfig) -> 'VocabManager':
        """从文件加载词表"""
        if load_path.endswith('.json'):
            with open(load_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
        else:
            with open(load_path, 'rb') as f:
                vocab_data = pickle.load(f)
        
        # 创建实例
        instance = cls(config)
        
        # 恢复状态
        instance.token_to_id = {int(k): v for k, v in vocab_data['token_to_id'].items()}
        instance.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        instance.vocab_size = vocab_data['vocab_size']
        instance.token_counts = Counter(vocab_data['token_counts'])
        instance._built = True
        
        print(f"词表已从 {load_path} 加载完成，词表大小: {instance.vocab_size}")
        return instance


def build_vocab_from_sequences(token_sequences: List[List[int]], 
                              config: ProjectConfig,
                              min_freq: int = 1,
                              max_vocab_size: Optional[int] = None) -> VocabManager:
    """从token序列构建词表的便捷函数
    
    Args:
        token_sequences: token ID序列列表
        config: 项目配置
        min_freq: 最小词频
        max_vocab_size: 最大词表大小
        
    Returns:
        构建好的VocabManager实例
    """
    print("开始从token序列构建词表...")
    
    vocab_manager = VocabManager(config)
    vocab_manager.add_token_sequences(token_sequences)
    vocab_manager.build_vocab(min_freq, max_vocab_size)
    
    return vocab_manager 