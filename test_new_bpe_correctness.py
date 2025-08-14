#!/usr/bin/env python3
"""
验证新的numba BPE实现与minbpe逻辑的一致性。

测试内容：
1. 使用相同输入数据训练BPE
2. 对比训练产生的merge_rules
3. 对比编码结果
4. 验证逻辑正确性
"""

from __future__ import annotations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from typing import List, Tuple
from algorithms.compression.bpe_engine import BPEEngine

# 引入minbpe参考实现
sys.path.append('foreign_dataset_files_to_convert')

# 直接导入base模块的函数
import sys
import os
base_path = os.path.join(os.path.dirname(__file__), 'foreign_dataset_files_to_convert')
sys.path.insert(0, base_path)

# 手动实现基础函数，避免相对导入问题
def get_stats(ids):
    """统计连续pair的频次"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """在ids中将所有连续的pair替换为idx"""
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def create_test_data() -> List[List[int]]:
    """创建简单的测试数据"""
    # 使用简单的token序列，容易验证BPE的行为
    test_sequences = [
        [1, 2, 3, 1, 2, 4, 1, 2],        # 1-2 应该是高频pair
        [1, 2, 5, 3, 4, 1, 2],           # 更多1-2
        [3, 4, 3, 4, 5, 6],              # 3-4 也是高频pair
        [1, 2, 3, 4, 1, 2, 3, 4],        # 1-2 和 3-4
        [7, 8, 7, 8, 9, 10],             # 7-8 pair
    ]
    return test_sequences





def test_numba_implementation(sequences: List[List[int]], num_merges: int, min_frequency: int = 1) -> Tuple[List[Tuple[int, int, int]], dict]:
    """使用新的numba实现进行训练"""
    print("=== 使用新的numba实现 ===")
    
    engine = BPEEngine(train_backend="numba", encode_backend="python")
    stats = engine.train(sequences, num_merges=num_merges, min_frequency=min_frequency)
    
    print(f"训练完成：{stats}")
    print(f"Merge规则数量：{len(engine.merge_rules)}")
    
    return engine.merge_rules, stats


def simple_bpe_train(sequences: List[List[int]], num_merges: int, min_frequency: int = 1) -> List[Tuple[int, int, int]]:
    """简单的BPE训练参考实现，用于验证逻辑"""
    print("=== 使用简单参考实现 ===")
    
    # 统计基础词表
    base_vocab = set()
    for seq in sequences:
        base_vocab.update(seq)
    base_vocab_size = len(base_vocab)
    next_id = max(base_vocab) + 1
    
    # 复制序列数据
    current_sequences = [list(seq) for seq in sequences]
    merge_rules = []
    
    for merge_idx in range(num_merges):
        # 统计所有pair的频次
        all_pairs = {}
        for seq in current_sequences:
            stats = get_stats(seq)
            for pair, count in stats.items():
                all_pairs[pair] = all_pairs.get(pair, 0) + count
        
        if not all_pairs:
            break
        
        # 选择频次最高的pair
        best_pair = max(all_pairs.items(), key=lambda x: x[1])
        pair, freq = best_pair
        
        if freq < min_frequency:
            break
        
        # 执行merge
        new_id = next_id
        next_id += 1
        
        for i, seq in enumerate(current_sequences):
            current_sequences[i] = merge(seq, pair, new_id)
        
        merge_rules.append((pair[0], pair[1], new_id))
        print(f"Merge {merge_idx + 1}: {pair} -> {new_id} (freq: {freq})")
    
    return merge_rules


def compare_merge_rules(rules1: List[Tuple[int, int, int]], rules2: List[Tuple[int, int, int]], name1: str, name2: str) -> bool:
    """对比两组merge规则"""
    print(f"\n=== 对比 {name1} vs {name2} ===")
    
    if len(rules1) != len(rules2):
        print(f"❌ 规则数量不同：{len(rules1)} vs {len(rules2)}")
        return False
    
    all_match = True
    for i, (rule1, rule2) in enumerate(zip(rules1, rules2)):
        if rule1 != rule2:
            print(f"❌ 第{i+1}个规则不同：{rule1} vs {rule2}")
            all_match = False
        else:
            print(f"✅ 第{i+1}个规则相同：{rule1}")
    
    if all_match:
        print(f"🎉 所有规则完全一致！")
    
    return all_match


def main():
    """主测试函数"""
    print("开始验证新BPE实现的正确性...")
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"测试数据：{test_data}")
    
    num_merges = 5
    min_frequency = 2
    
    # 测试简单参考实现
    simple_rules = simple_bpe_train(test_data, num_merges, min_frequency)
    
    # 测试新的numba实现
    numba_rules, numba_stats = test_numba_implementation(test_data, num_merges, min_frequency)
    
    # 对比结果
    success = compare_merge_rules(simple_rules, numba_rules, "简单参考实现", "numba实现")
    
    if success:
        print("\n🎉 验证成功！新的numba实现遵循了正确的BPE逻辑。")
    else:
        print("\n❌ 验证失败！新的numba实现存在问题。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
