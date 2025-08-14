#!/usr/bin/env python3
"""
测试修复后的BPE引擎
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from typing import List, Dict, Tuple

# 直接导入numba函数，避免复杂的包导入问题
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'algorithms', 'compression'))
from numba_bpe_train import count_pairs_ragged, apply_merge_and_delta_ragged

class FixedBPEEngine:
    """修复后的BPE引擎，直接实现修复逻辑"""
    
    def __init__(self):
        self.merge_rules: List[Tuple[int, int, int]] = []
        self.vocab_size: int = 0
    
    def train_with_fix(self, token_sequences: List[List[int]], *, num_merges: int, min_frequency: int) -> Dict:
        """使用修复逻辑的BPE训练"""
        print(f"🔧 开始修复后的BPE训练: num_merges={num_merges}, min_frequency={min_frequency}")
        
        # 统计基础词表
        base_vocab: Dict[int, None] = {}
        for s in token_sequences:
            for t in s:
                base_vocab[int(t)] = None
        base_vocab_size = len(base_vocab)
        print(f"📋 基础词表大小: {base_vocab_size}")
        
        # ragged表示
        offsets = [0]
        flat = []
        for s in token_sequences:
            flat.extend(int(x) for x in s)
            offsets.append(len(flat))
        flat = np.asarray(flat, dtype=np.int32)
        offsets = np.asarray(offsets, dtype=np.int32)
        
        # 初始pair统计
        k0, v0 = count_pairs_ragged(flat, offsets)
        global_counts: Dict[int, int] = {int(k0[i]): int(v0[i]) for i in range(len(k0))}
        
        print(f"📋 初始unique pairs: {len(global_counts)}")
        
        merges_done = 0
        next_id = base_vocab_size
        merge_rules: List[Tuple[int, int, int]] = []
        
        while merges_done < num_merges:
            # 选最佳pair
            best_key = None
            best_freq = 0
            for k, c in global_counts.items():
                if c >= min_frequency and c > best_freq:
                    best_key = k
                    best_freq = c
            
            if best_key is None:
                print(f"🔚 没有满足条件的pair，训练提前结束 (merges_done={merges_done})")
                break
                
            left_id = (best_key >> 32) & 0xFFFFFFFF
            right_id = best_key & 0xFFFFFFFF
            new_id = next_id
            next_id += 1
            
            print(f"🔄 第{merges_done+1}轮: 选择pair ({left_id}, {right_id}) -> {new_id}, 频次: {best_freq}")
            
            # 执行合并 - 这里是关键的修复逻辑
            new_flat, new_offsets, dk, dv = apply_merge_and_delta_ragged(
                flat, offsets, np.int32(left_id), np.int32(right_id), np.int32(new_id)
            )
            
            # 🔧 修复逻辑：检查是否有实际的合并发生
            if len(dk) == 0:
                print(f"⚠️  警告: pair ({left_id}, {right_id}) 在序列中不存在，从global_counts中移除")
                if best_key in global_counts:
                    del global_counts[best_key]
                continue  # 跳过这次合并，重新选择最佳pair
            
            # 有实际合并发生，更新状态
            flat, offsets = new_flat, new_offsets
            print(f"✅ 成功合并，flat长度: {len(flat)}, delta数量: {len(dk)}")
            
            # 应用delta
            for i in range(len(dk)):
                kk = int(dk[i])
                dvv = int(dv[i])
                newv = global_counts.get(kk, 0) + dvv
                if newv == 0:
                    if kk in global_counts:
                        del global_counts[kk]
                else:
                    global_counts[kk] = newv
            
            merge_rules.append((int(left_id), int(right_id), int(new_id)))
            merges_done += 1
        
        self.merge_rules = merge_rules
        self.vocab_size = base_vocab_size + merges_done
        
        print(f"🎯 训练完成: {merges_done} 次合并, 最终词表大小: {self.vocab_size}")
        return {"num_merges_performed": merges_done, "final_vocab_size": self.vocab_size}

def test_fixed_bpe():
    """测试修复后的BPE引擎"""
    
    # 使用和之前一样的测试数据
    test_sequences = []
    
    # 添加一些会产生(3,3)的序列
    for _ in range(20):
        test_sequences.append([3, 3, 3, 1, 2])
    
    # 添加一些会产生(13,13)的序列    
    for _ in range(15):
        test_sequences.append([13, 13, 1, 2])
    
    # 添加一些混合序列
    for _ in range(10):
        test_sequences.append([3, 3, 13, 13, 1, 2])
    
    # 添加一些包含更多token的序列
    for _ in range(5):
        test_sequences.append([3, 3, 3, 3, 1, 2, 3, 3])
    
    print(f"📋 测试数据: {len(test_sequences)} 个序列")
    print("🔧 测试修复后的BPE引擎...")
    
    # 创建修复后的BPE引擎
    engine = FixedBPEEngine()
    
    # 训练
    result = engine.train_with_fix(
        token_sequences=test_sequences,
        num_merges=50,  # 足够多的合并次数来测试
        min_frequency=3
    )
    
    print(f"\n📊 训练结果: {result}")
    print(f"📋 生成的merge规则数量: {len(engine.merge_rules)}")
    
    # 检查是否还有重复合并
    pair_counts = {}
    for left, right, new_id in engine.merge_rules:
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    print("\n🔍 重复合并检查:")
    has_duplicates = False
    for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            print(f"  ❌ {pair}: {count} 次 - 仍有重复!")
            has_duplicates = True
        else:
            print(f"  ✅ {pair}: {count} 次 - 正常")
    
    if not has_duplicates:
        print("\n🎉 成功！没有发现重复合并问题")
        print("✅ BPE引擎修复验证通过")
    else:
        print("\n😞 仍然存在重复合并问题")
        
    # 显示所有merge规则
    print(f"\n📋 所有 {len(engine.merge_rules)} 个merge规则:")
    for i, rule in enumerate(engine.merge_rules):
        print(f"  {i}: {rule}")
        if i >= 20:  # 只显示前20个避免输出过长
            print(f"  ... 还有 {len(engine.merge_rules) - 21} 个规则")
            break
    
    return not has_duplicates

if __name__ == "__main__":
    success = test_fixed_bpe()
    if success:
        print("\n🎯 BPE修复测试成功！")
    else:
        print("\n💥 BPE修复测试失败！")
    sys.exit(0 if success else 1)
