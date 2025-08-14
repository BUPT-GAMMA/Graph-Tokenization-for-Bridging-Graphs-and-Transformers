#!/usr/bin/env python3
"""
调试merge顺序问题
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'foreign_dataset_files_to_convert'))

from algorithms.compression.numba_bpe_train import count_pairs_single_seq, _unpack_pair
from int_basic_tokenizer import get_stats
import numpy as np


def debug_third_round():
    """调试第3轮的差异"""
    
    # 第2轮merge后的状态
    # 原始：[1, 2, 3, SEP, 3, 4, 5, SEP, 1, 2, 4, SEP, 4, 5, 6]
    # 第1轮(1,2)->8：[8, 3, SEP, 3, 4, 5, SEP, 8, 4, SEP, 4, 5, 6]  
    # 第2轮(4,5)->9：[8, 3, SEP, 3, 9, SEP, 8, 4, SEP, 9, 6]
    
    separator = 7  # 假设分隔符是7
    after_2nd_merge = [8, 3, 7, 3, 9, 7, 8, 4, 7, 9, 6]
    
    print("第2轮merge后的序列:")
    print(f"序列: {after_2nd_merge}")
    print(f"分隔符: {separator}")
    
    # 使用我们的numba实现统计
    seq_array = np.array(after_2nd_merge, dtype=np.int32)
    our_keys, our_counts = count_pairs_single_seq(seq_array, np.int32(separator))
    
    print("\n我们的pair统计:")
    pairs_our = []
    for i in range(len(our_keys)):
        left, right = _unpack_pair(np.int64(our_keys[i]))
        freq = int(our_counts[i])
        pairs_our.append(((left, right), freq))
        print(f"  ({left}, {right}): {freq}")
    
    # 使用minbpe的统计
    minbpe_stats = get_stats(after_2nd_merge, separator)
    
    print("\nminbpe的pair统计:")
    pairs_minbpe = []
    for pair, freq in minbpe_stats.items():
        pairs_minbpe.append((pair, freq))
        print(f"  {pair}: {freq}")
    
    # 排序并对比
    pairs_our_sorted = sorted(pairs_our, key=lambda x: (-x[1], x[0]))
    pairs_minbpe_sorted = sorted(pairs_minbpe.items(), key=lambda x: (-x[1], x[0]))
    
    print("\n排序后的对比:")
    print("我们的 (按频次降序):")
    for pair, freq in pairs_our_sorted:
        print(f"  {pair}: {freq}")
    
    print("minbpe的 (按频次降序):")
    for pair, freq in pairs_minbpe_sorted:
        print(f"  {pair}: {freq}")
    
    # 检查选择逻辑
    if pairs_our_sorted and pairs_minbpe_sorted:
        our_best = pairs_our_sorted[0]
        minbpe_best = pairs_minbpe_sorted[0]
        
        print(f"\n第3轮应该选择:")
        print(f"我们的选择: {our_best[0]} (freq: {our_best[1]})")
        print(f"minbpe选择: {minbpe_best[0]} (freq: {minbpe_best[1]})")
        
        if our_best != minbpe_best:
            print("❌ 选择不一致!")
            
            # 检查是否有频次相同的情况
            max_freq = max(pairs_our_sorted[0][1], pairs_minbpe_sorted[0][1])
            our_max_freq_pairs = [p for p in pairs_our_sorted if p[1] == max_freq]
            minbpe_max_freq_pairs = [p for p in pairs_minbpe_sorted if p[1] == max_freq]
            
            print(f"\n频次为{max_freq}的所有pairs:")
            print(f"我们的: {[p[0] for p in our_max_freq_pairs]}")
            print(f"minbpe: {[p[0] for p in minbpe_max_freq_pairs]}")
            
            if len(our_max_freq_pairs) > 1 or len(minbpe_max_freq_pairs) > 1:
                print("⚠️ 存在频次相同的pairs，可能是选择策略不同导致的")
        else:
            print("✅ 选择一致!")


if __name__ == "__main__":
    debug_third_round()
