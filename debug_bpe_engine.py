#!/usr/bin/env python3
"""
带调试信息的BPE引擎，用于定位重复合并问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from typing import List, Dict, Tuple

# 直接导入numba函数
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms', 'compression'))
from numba_bpe_train import count_pairs_ragged, apply_merge_and_delta_ragged

class DebugBPEEngine:
    """带详细调试信息的BPE引擎"""
    
    def __init__(self):
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size: int = 0
    
    def train(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        print(f"🔍 开始BPE训练调试: num_merges={num_merges}, min_frequency={min_frequency}")
        print(f"🔍 输入序列数量: {len(token_sequences)}")
        
        # 统计基础词表
        base_vocab: Dict[int, None] = {}
        for s in token_sequences:
            for t in s:
                base_vocab[int(t)] = None
        base_vocab_size = len(base_vocab)
        print(f"🔍 基础词表大小: {base_vocab_size}")
        
        # ragged表示
        offsets = [0]
        flat = []
        for s in token_sequences:
            flat.extend(int(x) for x in s)
            offsets.append(len(flat))
        flat = np.asarray(flat, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.int32)
        
        print(f"🔍 总token数: {len(flat)}")
        
        # 初始pair统计
        k0, v0 = count_pairs_ragged(flat, offsets)
        global_counts: Dict[int, int] = {int(k0[i]): int(v0[i]) for i in range(len(k0))}
        
        print(f"🔍 初始unique pair数量: {len(global_counts)}")
        print("🔍 初始top 10 pairs:")
        for i, (k, c) in enumerate(sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:10]):
            left_id = (k >> 32) & 0xFFFFFFFF
            right_id = k & 0xFFFFFFFF
            print(f"    {i}: ({left_id}, {right_id}) -> {c}")
        
        merges_done = 0
        next_id = base_vocab_size
        merge_rules: List[Tuple[int, int, int]] = []
        
        # 用于检测重复合并的历史记录
        merged_pairs: Dict[Tuple[int, int], List[int]] = {}
        
        while merges_done < num_merges:
            print(f"\n🔍 === 第 {merges_done + 1} 轮合并 ===")
            print(f"🔍 当前global_counts大小: {len(global_counts)}")
            
            # 选最佳pair - 添加详细调试
            best_key = None
            best_freq = 0
            candidates = []
            
            for k, c in global_counts.items():
                if c >= min_frequency:
                    candidates.append((k, c))
                    if c > best_freq:
                        best_key = k
                        best_freq = c
            
            print(f"🔍 候选pairs数量: {len(candidates)} (满足min_frequency={min_frequency})")
            
            if len(candidates) > 1:
                # 显示前几个候选
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                print("🔍 Top 5 候选pairs:")
                for i, (k, c) in enumerate(sorted_candidates[:5]):
                    left_id = (k >> 32) & 0xFFFFFFFF
                    right_id = k & 0xFFFFFFFF
                    mark = " ⭐" if k == best_key else ""
                    print(f"    {i}: ({left_id}, {right_id}) -> {c}{mark}")
            
            if best_key is None:
                print("🔍 没有找到满足条件的pair，停止训练")
                break
                
            left_id = (best_key >> 32) & 0xFFFFFFFF
            right_id = best_key & 0xFFFFFFFF
            new_id = next_id
            next_id += 1
            
            pair_tuple = (left_id, right_id)
            print(f"🔍 选择的最佳pair: {pair_tuple} -> {new_id}, 频次: {best_freq}")
            
            # 检查是否是重复合并
            if pair_tuple in merged_pairs:
                print(f"🚨 警告: pair {pair_tuple} 之前已经被合并过!")
                print(f"🚨 之前的合并: {merged_pairs[pair_tuple]}")
                print(f"🚨 这是第 {len(merged_pairs[pair_tuple]) + 1} 次合并同一个pair!")
                # 继续执行，看看会发生什么
            else:
                merged_pairs[pair_tuple] = []
            
            merged_pairs[pair_tuple].append(new_id)
            
            # 执行合并
            print(f"🔍 执行合并前: flat长度={len(flat)}")
            new_flat, new_offsets, dk, dv = apply_merge_and_delta_ragged(
                flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id)
            )
            print(f"🔍 执行合并后: flat长度={len(new_flat)}, 减少了 {len(flat) - len(new_flat)} 个token")
            
            flat, offsets = new_flat, new_offsets
            
            # 分析delta
            print(f"🔍 Delta变化项数: {len(dk)}")
            target_key_in_delta = False
            for i in range(len(dk)):
                kk = int(dk[i])
                dvv = int(dv[i])
                if kk == best_key:
                    target_key_in_delta = True
                    print(f"🔍 目标pair {pair_tuple} 在delta中: {dvv}")
            
            if not target_key_in_delta:
                print(f"🚨 警告: 目标pair {pair_tuple} 不在delta中!")
            
            # 应用delta
            old_global_counts = dict(global_counts)
            for i in range(len(dk)):
                kk = int(dk[i])
                dvv = int(dv[i])
                newv = global_counts.get(kk, 0) + dvv
                if newv == 0:
                    if kk in global_counts:
                        del global_counts[kk]
                else:
                    global_counts[kk] = newv
            
            # 检查目标pair是否被正确移除
            if best_key in global_counts:
                remaining = global_counts[best_key]
                print(f"🚨 严重错误: 目标pair {pair_tuple} 仍在global_counts中，剩余: {remaining}")
                print(f"🚨 原始频次: {best_freq}, 应该被完全移除")
                
                # 额外调试信息
                print("🔍 检查为什么没有被移除:")
                original_count = old_global_counts.get(best_key, 0)
                applied_delta = 0
                for i in range(len(dk)):
                    if int(dk[i]) == best_key:
                        applied_delta += int(dv[i])
                print(f"🔍   原始count: {original_count}")
                print(f"🔍   应用的delta: {applied_delta}")
                print(f"🔍   预期结果: {original_count + applied_delta}")
                print(f"🔍   实际结果: {global_counts[best_key]}")
            else:
                print(f"✅ 目标pair {pair_tuple} 已正确从global_counts中移除")
            
            merge_rules.append((int(left_id), int(right_id), int(new_id)))
            merges_done += 1
            
            # 显示当前top pairs
            if len(global_counts) > 0:
                print("🔍 更新后top 5 pairs:")
                for i, (k, c) in enumerate(sorted(global_counts.items(), key=lambda x: x[1], reverse=True)[:5]):
                    left = (k >> 32) & 0xFFFFFFFF
                    right = k & 0xFFFFFFFF
                    print(f"    {i}: ({left}, {right}) -> {c}")
        
        self.merge_rules = merge_rules
        self.vocab_size = base_vocab_size + merges_done
        
        print(f"\n🔍 === 训练完成 ===")
        print(f"🔍 总共执行: {merges_done} 次合并")
        print(f"🔍 最终词表大小: {self.vocab_size}")
        
        # 分析重复合并
        print("\n🔍 === 重复合并分析 ===")
        for pair, new_ids in merged_pairs.items():
            if len(new_ids) > 1:
                print(f"🚨 pair {pair} 被合并了 {len(new_ids)} 次: {new_ids}")
            
        return {"num_merges_performed": merges_done, "final_vocab_size": self.vocab_size}

def test_with_real_like_data():
    """使用更接近真实情况的测试数据"""
    # 创建一个更复杂的测试案例，模拟可能导致问题的情况
    test_sequences = []
    
    # 添加一些会产生(3,3)的序列
    for _ in range(20):
        test_sequences.append([3, 3, 3, 1, 2])
    
    # 添加一些会产生(13,13)的序列    
    for _ in range(15):
        test_sequences.append([13, 13, 1, 2])
    
    # 添加一些混合序列，可能产生复杂的相互作用
    for _ in range(10):
        test_sequences.append([3, 3, 13, 13, 1, 2])
    
    # 添加一些包含更多token的序列
    for _ in range(5):
        test_sequences.append([3, 3, 3, 3, 1, 2, 3, 3])
    
    print(f"📋 测试数据: {len(test_sequences)} 个序列")
    
    engine = DebugBPEEngine()
    result = engine.train(
        token_sequences=test_sequences,
        num_merges=50,  # 更多合并次数
        min_frequency=3  # 降低最小频次
    )
    
    print(f"\n📊 最终结果: {result}")
    print("\n📋 所有merge规则:")
    for i, rule in enumerate(engine.merge_rules):
        print(f"  {i}: {rule}")

if __name__ == "__main__":
    test_with_real_like_data()
