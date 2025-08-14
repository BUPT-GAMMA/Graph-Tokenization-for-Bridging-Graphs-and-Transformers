#!/usr/bin/env python3
"""
调试BPE训练中的重复合并问题
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 直接导入需要的模块，避免C++后端问题
from algorithms.compression.bpe_engine import BPEEngine

def debug_bpe_training():
    """用简单的测试数据调试BPE训练"""
    
    # 创建一个简单的测试案例，能复现 (5, 3) 重复合并的问题
    # 假设我们有一些简单的token序列
    test_sequences = [
        [3, 3, 3, 1, 2],  # 这会产生 (3,3) 对，合并后可能产生 (5,3) 
        [3, 3, 3, 1, 2],
        [3, 3, 3, 1, 2],
        [3, 3, 3, 1, 2],
        [3, 3, 3, 1, 2],
        [13, 13, 1, 2],   # 这会产生 (13,13) 对
        [13, 13, 1, 2],
        [13, 13, 1, 2],
        [13, 13, 1, 2],
    ]
    
    print("原始测试序列:")
    for i, seq in enumerate(test_sequences):
        print(f"  {i}: {seq}")
    
    # 创建BPE引擎
    engine = BPEEngine(train_backend="numba")
    
    # 训练
    print("\n开始BPE训练...")
    result = engine.train(
        token_sequences=test_sequences,
        num_merges=10,  # 只做10次合并来观察问题
        min_frequency=2
    )
    
    print(f"\n训练结果: {result}")
    print(f"\nMerge rules (前20个):")
    for i, rule in enumerate(engine.merge_rules[:20]):
        print(f"  {i}: {rule}")
        
    # 分析问题
    print(f"\n总共生成了 {len(engine.merge_rules)} 个merge规则")
    
    # 统计每个对出现的次数
    pair_counts = {}
    for left, right, new_id in engine.merge_rules:
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    
    print("\n每个对被合并的次数:")
    for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:
            print(f"  {pair}: {count} 次 - 这是错误的！同一对不应该被合并多次")
        else:
            print(f"  {pair}: {count} 次 - 正常")

if __name__ == "__main__":
    debug_bpe_training()
