#!/usr/bin/env python3
"""
大规模数据集BPE严格对拍测试
专门用于测试zinc等大规模数据集，仅在异常时输出详细信息
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'foreign_dataset_files_to_convert'))

from typing import List, Tuple, Dict, Any
import numpy as np
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.compression.bpe_engine import BPEEngine
from foreign_dataset_files_to_convert.int_basic_tokenizer import IntBasicTokenizer

def test_large_scale_bpe(dataset_name: str = "zinc", num_merges: int = 50, min_frequency: int = 2):
    """
    大规模数据集BPE测试
    
    Args:
        dataset_name: 数据集名称 
        num_merges: merge次数
        min_frequency: 最小频次
    """
    print(f"🚀 开始大规模BPE测试: {dataset_name}")
    print(f"参数: num_merges={num_merges}, min_frequency={min_frequency}")
    
    try:
        # 1. 加载数据
        print(f"📂 加载 {dataset_name} 数据...")
        
        # 创建配置和数据接口
        config = ProjectConfig()
        data_interface = UnifiedDataInterface(
            config=config,
            dataset=dataset_name
        )
        
        # 获取所有数据（使用get_sequences_by_splits然后合并）
        train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = data_interface.get_sequences_by_splits("dfs")
        
        # 提取纯序列数据（去掉图ID）
        all_sequences_with_ids = train_seqs + val_seqs + test_seqs
        sequences = [seq for graph_id, seq in all_sequences_with_ids]
        
        print(f"数据统计:")
        print(f"  序列数量: {len(sequences)}")
        total_tokens = sum(len(seq) for seq in sequences)
        print(f"  总token数: {total_tokens}")
        print(f"  平均序列长度: {total_tokens / len(sequences):.1f}")
        
        # 2. 训练minbpe参考实现
        print("🔍 训练minbpe参考实现...")
        minbpe_tokenizer = IntBasicTokenizer()
        minbpe_stats = minbpe_tokenizer.train(
            sequences, 
            num_merges=num_merges, 
            min_frequency=min_frequency,
            verbose=False  # 大规模数据不详细输出
        )
        minbpe_rules = [(left, right, new_id) for (left, right), new_id in minbpe_tokenizer.merges.items()]
        
        # 3. 训练我们的numba实现
        print("🔍 训练numba实现...")
        our_engine = BPEEngine(train_backend="numba", encode_backend="python")
        our_stats = our_engine.train(
            sequences, 
            num_merges=num_merges, 
            min_frequency=min_frequency
        )
        our_rules = our_engine.merge_rules
        
        # 4. 对比训练结果
        print("📊 对比训练结果...")
        
        # 检查merge数量
        minbpe_merges = minbpe_stats['num_merges_performed']
        our_merges = our_stats['num_merges_performed']
        
        if minbpe_merges != our_merges:
            print(f"❌ MERGE数量不一致!")
            print(f"  minbpe: {minbpe_merges} merges")
            print(f"  我们的: {our_merges} merges")
            return False
            
        # 逐步对比merge规则
        print(f"📋 对比 {minbpe_merges} 个merge规则...")
        for i in range(min(len(minbpe_rules), len(our_rules))):
            minbpe_rule = (minbpe_rules[i][0], minbpe_rules[i][1])
            our_rule = (our_rules[i][0], our_rules[i][1])
            
            if minbpe_rule != our_rule:
                print(f"❌ 第{i+1}轮规则不一致!")
                print(f"  minbpe: {minbpe_rule}")
                print(f"  我们的: {our_rule}")
                
                # 输出前后几轮的规则用于调试
                start_idx = max(0, i-2)
                end_idx = min(len(minbpe_rules), i+3)
                print(f"\n📋 周围规则对比 (第{start_idx+1}-{end_idx}轮):")
                for j in range(start_idx, end_idx):
                    if j < len(minbpe_rules) and j < len(our_rules):
                        mb_r = (minbpe_rules[j][0], minbpe_rules[j][1])
                        our_r = (our_rules[j][0], our_rules[j][1])
                        status = "✅" if mb_r == our_r else "❌"
                        print(f"  第{j+1}轮: {status} minbpe:{mb_r} vs 我们的:{our_r}")
                return False
        
        # 5. 对比编码结果（抽样检查）
        print("🔧 抽样检查编码结果...")
        our_engine.build_encoder()
        
        # 抽样检查前100个序列
        sample_size = min(100, len(sequences))
        for i in range(sample_size):
            seq = sequences[i]
            
            minbpe_encoded = minbpe_tokenizer.encode(seq)
            our_encoded = our_engine.encode(seq)
            
            if minbpe_encoded != our_encoded:
                print(f"❌ 序列{i+1}编码不一致!")
                print(f"  原始: {seq}")
                print(f"  minbpe: {minbpe_encoded}")
                print(f"  我们的: {our_encoded}")
                return False
        
        print(f"✅ 大规模测试通过!")
        print(f"  数据集: {dataset_name}")
        print(f"  序列数: {len(sequences)}")
        print(f"  merge数: {minbpe_merges}")
        print(f"  抽样编码: {sample_size}/{len(sequences)} 通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试zinc数据集
    success = test_large_scale_bpe("zinc", num_merges=50, min_frequency=2)
    
    if success:
        print("\n🎉 zinc数据集测试完全成功!")
    else:
        print("\n💥 zinc数据集测试失败!")
        sys.exit(1)
