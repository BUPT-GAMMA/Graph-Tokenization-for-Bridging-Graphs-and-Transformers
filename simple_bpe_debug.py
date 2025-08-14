#!/usr/bin/env python3
"""
简化的BPE调试脚本，直接测试numba训练逻辑
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from typing import List, Dict, Tuple

# 直接导入numba函数，避免包初始化问题
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms', 'compression'))
from numba_bpe_train import count_pairs_ragged, apply_merge_and_delta_ragged

def simple_bpe_debug():
    """直接使用numba函数调试BPE逻辑"""
    
    # 创建简单的测试数据
    test_sequences = [
        [3, 3, 3, 1, 2],  # 这会产生 (3,3) 对
        [3, 3, 3, 1, 2],
        [3, 3, 3, 1, 2],
        [3, 3, 3, 1, 2],
        [13, 13, 1, 2],   # 这会产生 (13,13) 对
        [13, 13, 1, 2],
        [13, 13, 1, 2],
    ]
    
    print("原始测试序列:")
    for i, seq in enumerate(test_sequences):
        print(f"  {i}: {seq}")
    
    # 转换为ragged格式
    offsets = [0]
    flat = []
    for s in test_sequences:
        flat.extend(int(x) for x in s)
        offsets.append(len(flat))
    flat = np.asarray(flat, dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int32)
    
    print(f"\nRagged格式:")
    print(f"  flat: {flat}")
    print(f"  offsets: {offsets}")
    
    # 统计基础词表
    base_vocab: Dict[int, None] = {}
    for s in test_sequences:
        for t in s:
            base_vocab[int(t)] = None
    base_vocab_size = len(base_vocab)
    print(f"\n基础词表大小: {base_vocab_size}")
    print(f"基础词表: {list(base_vocab.keys())}")
    
    # 初始pair统计
    print("\n=== 初始pair统计 ===")
    k0, v0 = count_pairs_ragged(flat, offsets)
    global_counts: Dict[int, int] = {int(k0[i]): int(v0[i]) for i in range(len(k0))}
    
    print("初始global_counts:")
    for k, c in sorted(global_counts.items(), key=lambda x: x[1], reverse=True):
        left_id = (k >> 32) & 0xFFFFFFFF
        right_id = k & 0xFFFFFFFF
        print(f"  ({left_id}, {right_id}): {c}")
    
    # 模拟BPE训练过程
    merges_done = 0
    next_id = base_vocab_size
    merge_rules: List[Tuple[int, int, int]] = []
    num_merges = 10
    min_frequency = 2
    
    print(f"\n=== 开始BPE训练 (最多{num_merges}次合并, 最小频次{min_frequency}) ===")
    
    while merges_done < num_merges:
        print(f"\n--- 第 {merges_done + 1} 轮合并 ---")
        
        # 选最佳pair
        best_key = None
        best_freq = 0
        for k, c in global_counts.items():
            if c >= min_frequency and c > best_freq:
                best_key = k
                best_freq = c
        
        if best_key is None:
            print("没有找到满足最小频次要求的pair，停止训练")
            break
            
        left_id = (best_key >> 32) & 0xFFFFFFFF
        right_id = best_key & 0xFFFFFFFF
        new_id = next_id
        next_id += 1
        
        print(f"选择最佳pair: ({left_id}, {right_id}) -> {new_id}, 频次: {best_freq}")
        
        # 执行合并
        new_flat, new_offsets, dk, dv = apply_merge_and_delta_ragged(
            flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id)
        )
        
        print(f"合并前flat长度: {len(flat)}, 合并后: {len(new_flat)}")
        print(f"Delta变化数量: {len(dk)}")
        
        # 更新flat和offsets
        flat, offsets = new_flat, new_offsets
        
        # 打印delta详情
        print("Delta详情:")
        for i in range(len(dk)):
            kk = int(dk[i])
            dvv = int(dv[i])
            left = (kk >> 32) & 0xFFFFFFFF
            right = kk & 0xFFFFFFFF
            print(f"  ({left}, {right}): {dvv:+d}")
        
        # 应用delta到global_counts
        old_counts = dict(global_counts)  # 保存旧的计数用于比较
        for i in range(len(dk)):
            kk = int(dk[i])
            dvv = int(dv[i])
            newv = global_counts.get(kk, 0) + dvv
            if newv == 0:
                if kk in global_counts:
                    del global_counts[kk]
            else:
                global_counts[kk] = newv
        
        print("更新后的global_counts (只显示前10个最频繁的):")
        for i, (k, c) in enumerate(sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            left = (k >> 32) & 0xFFFFFFFF
            right = k & 0xFFFFFFFF
            old_c = old_counts.get(k, 0)
            change = c - old_c
            print(f"  ({left}, {right}): {c} (变化: {change:+d})")
        
        # 检查刚刚合并的pair是否还在counts中
        if best_key in global_counts:
            remaining = global_counts[best_key]
            print(f"⚠️  警告: 刚刚合并的pair ({left_id}, {right_id}) 仍然在counts中，剩余: {remaining}")
        else:
            print(f"✓ 刚刚合并的pair ({left_id}, {right_id}) 已从counts中移除")
        
        merge_rules.append((int(left_id), int(right_id), int(new_id)))
        merges_done += 1
    
    print(f"\n=== 训练完成 ===")
    print(f"总共执行了 {merges_done} 次合并")
    print("Merge rules:")
    for i, rule in enumerate(merge_rules):
        print(f"  {i}: {rule}")
    
    # 检查重复合并
    pair_counts = {}
    for left, right, new_id in merge_rules:
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    print("\n每个pair被合并的次数:")
    has_duplicates = False
    for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            print(f"  {pair}: {count} 次 - ❌ 错误！同一对不应该被合并多次")
            has_duplicates = True
        else:
            print(f"  {pair}: {count} 次 - ✓ 正常")
    
    if has_duplicates:
        print("\n🚨 发现重复合并问题！")
    else:
        print("\n✅ 没有发现重复合并问题")

if __name__ == "__main__":
    simple_bpe_debug()
