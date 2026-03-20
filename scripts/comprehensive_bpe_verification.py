#!/usr/bin/env python3
"""对拍验证：minBPE参考 vs 我们的引擎（规则+全量编码一致性）"""
#!/usr/bin/env python3
"""
BPE 全面对拍验证程序
- 对比 minBPE 参考实现与我们的引擎：
  1) 训练阶段：merge 规则完全一致
  2) 编码阶段：全量序列编码结果完全一致
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# 添加路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'foreign_dataset_files_to_convert'))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.compression.bpe_engine import BPEEngine
from foreign_dataset_files_to_convert.int_basic_tokenizer import IntBasicTokenizer

def load_test_sequences(dataset: str = "qm9test", limit: int = None) -> List[List[int]]:
    """加载测试序列"""
    print(f"📂 加载 {dataset} 数据集...")
    
    config = ProjectConfig()
    config.dataset.name = dataset
    
    udi = UnifiedDataInterface(config=config, dataset=dataset)
    sequences_with_ids, _ = udi.get_sequences("feuler")
    sequences = [seq for _, seq in sequences_with_ids]
    
    if limit and len(sequences) > limit:
        sequences = sequences[:limit]
    
    # 确保所有元素都是int类型
    sequences = [[int(token) for token in seq] for seq in sequences]
    
    print(f"✅ 数据加载完成：{len(sequences)}个序列，总计{sum(len(s) for s in sequences)}个token")
    return sequences

def train_minbpe_reference(sequences: List[List[int]], num_merges: int, min_frequency: int) -> Tuple[Dict, List]:
    """训练minbpe参考实现"""
    print("🔍 训练minbpe参考实现...")
    
    tokenizer = IntBasicTokenizer()
    start_time = time.perf_counter()
    
    stats = tokenizer.train(
        sequences,
        num_merges=num_merges,
        min_frequency=min_frequency,
        verbose=False
    )
    
    end_time = time.perf_counter()
    
    # 提取merge规则
    merge_rules = []
    for (left, right), new_id in tokenizer.merges.items():
        merge_rules.append((left, right, new_id))
    
    print(f"✅ minbpe训练完成：{end_time - start_time:.3f}秒，{len(merge_rules)}个merge规则")
    
    return {
        'stats': stats,
        'merge_rules': merge_rules,
        'vocab_size': len(tokenizer.vocab),
        'training_time': end_time - start_time
    }, tokenizer

def train_our_engine(sequences: List[List[int]], num_merges: int, min_frequency: int,
                     *, our_train_backend: str = "cpp", our_encode_backend: str = "cpp") -> Tuple[Dict, BPEEngine]:
    """训练我们的BPE引擎（可选择后端）。"""
    print(f"🚀 训练我们的BPE引擎... (train_backend={our_train_backend}, encode_backend={our_encode_backend})")
    
    engine = BPEEngine(train_backend=our_train_backend, encode_backend=our_encode_backend)
    start_time = time.perf_counter()
    
    stats = engine.train(
        sequences,
        num_merges=num_merges,
        min_frequency=min_frequency
    )
    
    end_time = time.perf_counter()
    
    print(f"✅ 我们的引擎训练完成：{end_time - start_time:.3f}秒，{len(engine.merge_rules)}个merge规则")
    
    return {
        'stats': stats,
        'merge_rules': engine.merge_rules,
        'vocab_size': engine.vocab_size,
        'training_time': end_time - start_time
    }, engine

def compare_merge_rules(minbpe_rules: List[Tuple], our_rules: List[Tuple]) -> bool:
    """详细对比merge规则"""
    print("📋 对比merge规则...")
    
    min_len = min(len(minbpe_rules), len(our_rules))
    max_len = max(len(minbpe_rules), len(our_rules))
    
    print(f"  minbpe规则数：{len(minbpe_rules)}")
    print(f"  我们的规则数：{len(our_rules)}")
    
    if len(minbpe_rules) != len(our_rules):
        print(f"❌ 规则数量不一致！")
        return False
    
    # 逐条对比
    all_match = True
    for i in range(min_len):
        minbpe_rule = (minbpe_rules[i][0], minbpe_rules[i][1])  # 只比较left, right
        our_rule = (our_rules[i][0], our_rules[i][1])
        
        if minbpe_rule != our_rule:
            print(f"❌ 第{i+1}条规则不一致：")
            print(f"    minbpe: {minbpe_rule}")
            print(f"    我们的: {our_rule}")
            
            # 显示前后几条规则的上下文
            context_start = max(0, i-2)
            context_end = min(min_len, i+3)
            print(f"    上下文对比 (第{context_start+1}-{context_end}条):")
            for j in range(context_start, context_end):
                if j < len(minbpe_rules) and j < len(our_rules):
                    mb_r = (minbpe_rules[j][0], minbpe_rules[j][1])
                    our_r = (our_rules[j][0], our_rules[j][1])
                    status = "✅" if mb_r == our_r else "❌"
                    print(f"      {j+1}: {status} minbpe:{mb_r} vs 我们:{our_r}")
            
            all_match = False
            break
    
    if all_match:
        print("✅ 所有merge规则完全一致！")
    
    return all_match

def compare_encoding_results(sequences: List[List[int]], minbpe_tokenizer, our_engine, sample_size: int = 1000) -> bool:
    """对比编码结果"""
    print(f"🔧 对比编码结果（抽样{sample_size}个序列）...")
    
    # 确保我们的引擎已构建编码器
    our_engine.build_encoder()
    
    sample_sequences = sequences[:sample_size] if len(sequences) > sample_size else sequences
    all_match = True
    
    for i, seq in enumerate(sample_sequences):
        minbpe_encoded = minbpe_tokenizer.encode(seq)
        our_encoded = our_engine.encode(seq)
        
        if minbpe_encoded != our_encoded:
            print(f"❌ 第{i+1}个序列编码不一致：")
            print(f"    原始序列: {seq[:20]}{'...' if len(seq) > 20 else ''}")
            print(f"    minbpe编码: {minbpe_encoded}")
            print(f"    我们的编码: {our_encoded}")
            all_match = False
            
            # 只显示前几个不一致的情况
            if not all_match and i >= 5:
                print("    （仅显示前5个不一致的情况）")
                break
    
    if all_match:
        print(f"✅ 所有{len(sample_sequences)}个序列的编码结果完全一致！")
    
    return all_match

def comprehensive_verification(dataset: str = "qm9test", num_merges: int = 2000, min_frequency: int = 100, 
                             encoding_sample_size: int = 1000, sequence_limit: int = None,
                             *, our_train_backend: str = "numba", our_encode_backend: str = "cpp"):
    """全面验证程序"""
    
    print("=" * 80)
    print("🎯 BPE全面对拍验证")
    print("=" * 80)
    print(f"数据集: {dataset}")
    print(f"merge数量: {num_merges}")
    print(f"最小频率: {min_frequency}")
    print(f"编码抽样: {encoding_sample_size}")
    if sequence_limit:
        print(f"序列限制: {sequence_limit}")
    print()
    
    # 1. 加载测试数据
    sequences = load_test_sequences(dataset, limit=sequence_limit)
    print()
    
    # 2. 训练两个实现
    try:
        minbpe_result, minbpe_tokenizer = train_minbpe_reference(sequences, num_merges, min_frequency)
        our_result, our_engine = train_our_engine(sequences, num_merges, min_frequency,
                                                 our_train_backend=our_train_backend,
                                                 our_encode_backend=our_encode_backend)
        print()
        
        # 3. 对比训练结果
        print("📊 训练结果对比：")
        print(f"  minbpe: {minbpe_result['stats']['num_merges_performed']} merges, "
              f"{minbpe_result['vocab_size']} vocab, {minbpe_result['training_time']:.3f}s")
        print(f"  我们的: {our_result['stats']['num_merges_performed']} merges, "
              f"{our_result['vocab_size']} vocab, {our_result['training_time']:.3f}s")
        print()
        
        # 4. 详细对比merge规则
        rules_match = compare_merge_rules(minbpe_result['merge_rules'], our_result['merge_rules'])
        print()
        
        # 5. 对比编码结果
        if rules_match:
            encoding_match = compare_encoding_results(sequences, minbpe_tokenizer, our_engine, encoding_sample_size)
        else:
            print("⚠️  由于merge规则不一致，跳过编码对比")
            encoding_match = False
        print()
        
        # 6. 最终结论
        print("=" * 80)
        print("📋 验证总结")
        print("=" * 80)
        
        if rules_match and encoding_match:
            print("🎉 完全验证通过！")
            print("✅ merge规则完全一致")
            print("✅ 编码结果完全一致")
            print("✅ 我们的实现与minbpe完全等价")
            return True
        else:
            print("💥 验证失败！")
            if not rules_match:
                print("❌ merge规则不一致")
            if not encoding_match:
                print("❌ 编码结果不一致")
            print("❗ 需要修复逻辑差异")
            return False
            
    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BPE全面对拍验证")
    parser.add_argument("--dataset", default="qm9test", choices=["qm9test", "qm9", "zinc", "aqsol"],
                       help="测试数据集")
    parser.add_argument("--num-merges", type=int, default=2000, help="merge操作数量")
    parser.add_argument("--min-frequency", type=int, default=100, help="最小频次阈值")
    parser.add_argument("--encoding-sample", type=int, default=1000, help="编码验证抽样数量")
    parser.add_argument("--limit", type=int, help="限制序列数量（用于快速测试）")
    parser.add_argument("--our-train-backend", choices=["python","cpp"], default="cpp", help="我们的引擎训练后端")
    parser.add_argument("--our-encode-backend", choices=["cpp","python"], default="cpp", help="我们的引擎编码后端")
    
    args = parser.parse_args()
    
    success = comprehensive_verification(
        dataset=args.dataset,
        num_merges=args.num_merges,
        min_frequency=args.min_frequency,
        encoding_sample_size=args.encoding_sample,
        sequence_limit=args.limit,
        our_train_backend=args.our_train_backend,
        our_encode_backend=args.our_encode_backend
    )
    
    if success:
        print("\n🎉 验证成功！可以进行性能优化了。")
        sys.exit(0)
    else:
        print("\n💥 验证失败！需要先修复逻辑问题。")
        sys.exit(1)
