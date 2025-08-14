#!/usr/bin/env python3
"""
调试分隔符处理的差异问题
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'foreign_dataset_files_to_convert'))

from algorithms.compression.numba_bpe_train import count_pairs_single_seq, _unpack_pair, _pack_pair
from int_basic_tokenizer import get_stats
import numpy as np


def debug_bpe_training():
    """详细调试BPE训练过程"""
    
    test_sequences = [
        [1, 2, 3],
        [3, 4, 5], 
        [1, 2, 4],
        [4, 5, 6],
    ]
    
    print("🔍 调试分隔符处理...")
    print(f"测试序列: {test_sequences}")
    
    # 计算分隔符
    all_tokens = []
    for seq in test_sequences:
        all_tokens.extend(seq)
    max_base_id = max(all_tokens)
    separator_token = max_base_id + 1
    
    print(f"max_base_id: {max_base_id}")
    print(f"separator_token: {separator_token}")
    
    # 构建带分隔符的序列
    combined_sequence = []
    for i, seq in enumerate(test_sequences):
        if i > 0:
            combined_sequence.append(separator_token)
        combined_sequence.extend(seq)
    
    print(f"拼接后的序列: {combined_sequence}")
    
    # 模拟前两轮merge
    current_seq = np.array(combined_sequence, dtype=np.int32)
    next_id = separator_token + 1
    
    print("\n第1轮merge前的状态:")
    print(f"序列: {current_seq.tolist()}")
    
    # 第1轮：统计pairs
    pair_keys, pair_counts = count_pairs_single_seq(current_seq)
    print("所有pairs及其频次:")
    for i in range(len(pair_keys)):
        left, right = _unpack_pair(np.int64(pair_keys[i]))
        freq = int(pair_counts[i])
        print(f"  ({left}, {right}): {freq}")
    
    # 选择最佳pair (1,2) -> 8
    current_seq = apply_manual_merge(current_seq, (1, 2), next_id)
    next_id += 1
    print(f"\n第1轮merge后: {current_seq.tolist()}")
    
    # 第2轮：统计pairs
    pair_keys, pair_counts = count_pairs_single_seq(current_seq)
    print("\n第2轮merge前的pairs:")
    for i in range(len(pair_keys)):
        left, right = _unpack_pair(np.int64(pair_keys[i]))
        freq = int(pair_counts[i])
        print(f"  ({left}, {right}): {freq}")
        
    # 选择最佳pair (4,5) -> 9
    current_seq = apply_manual_merge(current_seq, (4, 5), next_id)
    next_id += 1
    print(f"\n第2轮merge后: {current_seq.tolist()}")
    
    # 第3轮：统计pairs
    pair_keys, pair_counts = count_pairs_single_seq(current_seq)
    print("\n第3轮merge前的pairs:")
    for i in range(len(pair_keys)):
        left, right = _unpack_pair(np.int64(pair_keys[i]))
        freq = int(pair_counts[i])
        print(f"  ({left}, {right}): {freq}")
    
    # 比较minbpe的做法
    print("\n" + "="*50)
    print("对比minbpe的处理...")
    
    # 用minbpe的get_stats
    minbpe_stats = get_stats(combined_sequence)
    print("minbpe第1轮前的pairs:")
    for pair, freq in sorted(minbpe_stats.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq}")


def apply_manual_merge(seq, target_pair, new_id):
    """手动应用merge操作"""
    result = []
    i = 0
    while i < len(seq):
        if (i + 1 < len(seq) and 
            seq[i] == target_pair[0] and 
            seq[i + 1] == target_pair[1]):
            result.append(new_id)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return np.array(result, dtype=np.int32)


def manual_merge_for_minbpe(ids, pair, new_id):
    """minbpe风格的merge"""
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(new_id)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


if __name__ == "__main__":
    debug_bpe_training()
