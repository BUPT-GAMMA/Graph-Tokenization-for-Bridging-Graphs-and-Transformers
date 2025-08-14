#!/usr/bin/env python3
"""
测试分隔符修复的有效性

验证：
1. 不会产生跨序列的错误pair合并
2. 分隔符被正确处理
3. 修复后的结果依然与minbpe一致
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'foreign_dataset_files_to_convert'))

from algorithms.compression.bpe_engine import BPEEngine
from int_basic_tokenizer import IntBasicTokenizer


def test_separator_effectiveness():
    """测试分隔符的有效性"""
    print("🔍 测试分隔符修复的有效性...")
    
    # 设计一个容易产生跨序列错误合并的测试用例
    test_sequences = [
        [1, 2, 3],      # 序列1: 结尾是3
        [3, 4, 5],      # 序列2: 开头是3（如果没有分隔符，会错误地合并(3,3)）
        [1, 2, 4],      # 序列3: 包含1,2和4
        [4, 5, 6],      # 序列4: 开头是4（如果没有分隔符，会错误地合并(4,4)）
    ]
    
    print(f"测试序列: {test_sequences}")
    print("如果没有分隔符，可能会错误产生:")
    print("- (3, 3): 从序列1结尾的3和序列2开头的3")
    print("- (4, 4): 从序列3结尾的4和序列4开头的4")
    
    # 测试我们的实现
    print("\n🔍 测试我们的numba实现...")
    our_engine = BPEEngine(train_backend="numba", encode_backend="python")
    our_stats = our_engine.train(test_sequences, num_merges=5, min_frequency=1)
    
    print("我们的merge规则:")
    for i, (left, right, new_id) in enumerate(our_engine.merge_rules):
        print(f"  第{i+1}轮: ({left}, {right}) -> {new_id}")
    
    # 测试minbpe参考实现
    print("\n🔍 测试minbpe参考实现...")
    minbpe_tokenizer = IntBasicTokenizer()
    minbpe_stats = minbpe_tokenizer.train(test_sequences, num_merges=5, min_frequency=1, verbose=False)
    
    print("minbpe的merge规则:")
    for i, (left, right, new_id) in enumerate(minbpe_tokenizer.get_merge_rules()):
        print(f"  第{i+1}轮: ({left}, {right}) -> {new_id}")
    
    # 对比结果
    our_rules = our_engine.merge_rules
    minbpe_rules = minbpe_tokenizer.get_merge_rules()
    
    print("\n📊 对比结果:")
    if len(our_rules) != len(minbpe_rules):
        print(f"❌ 规则数量不同: {len(our_rules)} vs {len(minbpe_rules)}")
        return False
    
    all_match = True
    for i, (our_rule, minbpe_rule) in enumerate(zip(our_rules, minbpe_rules)):
        our_pair = (our_rule[0], our_rule[1])
        minbpe_pair = (minbpe_rule[0], minbpe_rule[1])
        
        if our_pair != minbpe_pair:
            print(f"❌ 第{i+1}轮规则不同: {our_pair} vs {minbpe_pair}")
            all_match = False
        else:
            print(f"✅ 第{i+1}轮规则一致: {our_pair}")
    
    # 检查是否有跨序列的错误合并
    print("\n🔍 检查是否存在跨序列的错误合并...")
    problematic_pairs = [(3, 3), (4, 4)]  # 这些pairs不应该出现
    
    found_problematic = False
    for left, right, _ in our_rules:
        if (left, right) in problematic_pairs:
            print(f"❌ 发现跨序列的错误合并: ({left}, {right})")
            found_problematic = True
    
    if not found_problematic:
        print("✅ 没有发现跨序列的错误合并")
    
    return all_match and not found_problematic


def test_encoding_consistency():
    """测试编码结果的一致性"""
    print("\n🔧 测试编码结果的一致性...")
    
    test_sequences = [
        [1, 2, 3],
        [3, 4, 5],
        [1, 2, 4],
        [4, 5, 6],
    ]
    
    # 训练两个实现
    our_engine = BPEEngine(train_backend="numba", encode_backend="python")
    our_engine.train(test_sequences, num_merges=3, min_frequency=1)
    our_engine.build_encoder()
    
    minbpe_tokenizer = IntBasicTokenizer()
    minbpe_tokenizer.train(test_sequences, num_merges=3, min_frequency=1)
    
    # 测试每个序列的编码
    all_match = True
    for i, seq in enumerate(test_sequences):
        our_encoded = our_engine.encode(seq)
        minbpe_encoded = minbpe_tokenizer.encode(seq)
        
        if our_encoded != minbpe_encoded:
            print(f"❌ 序列{i+1}编码不一致:")
            print(f"  原始: {seq}")
            print(f"  我们的: {our_encoded}")
            print(f"  minbpe: {minbpe_encoded}")
            all_match = False
        else:
            print(f"✅ 序列{i+1}编码一致: {seq} -> {our_encoded}")
    
    return all_match


def main():
    """主测试函数"""
    print("🚀 开始测试分隔符修复...")
    
    # 测试1: 分隔符有效性
    separator_ok = test_separator_effectiveness()
    
    # 测试2: 编码一致性
    encoding_ok = test_encoding_consistency()
    
    # 总结
    print("\n" + "="*50)
    print("测试总结:")
    print(f"分隔符有效性: {'✅ 通过' if separator_ok else '❌ 失败'}")
    print(f"编码一致性: {'✅ 通过' if encoding_ok else '❌ 失败'}")
    
    if separator_ok and encoding_ok:
        print("\n🎉 分隔符修复成功！所有测试通过！")
        return True
    else:
        print("\n❌ 分隔符修复存在问题，需要进一步检查。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
