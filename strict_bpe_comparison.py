#!/usr/bin/env python3
"""
严格的BPE对拍测试脚本

功能：
1. 对比训练过程中每一轮的选择
2. 对比最终的merge rules
3. 对比encode结果
4. 支持多种数据集测试
5. 提供详细的差异分析

确保我们的实现与minbpe逻辑严格一致
"""

from __future__ import annotations
import sys
import os
import json
from typing import List, Tuple, Dict, Any
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'foreign_dataset_files_to_convert'))

from algorithms.compression.bpe_engine import BPEEngine
from int_basic_tokenizer import IntBasicTokenizer


class StrictBPEComparison:
    """严格的BPE对拍比较器"""
    
    def __init__(self):
        self.results = {}
        
    def compare_training(self, 
                        token_sequences: List[List[int]], 
                        num_merges: int, 
                        min_frequency: int = 1,
                        test_name: str = "default") -> Dict[str, Any]:
        """
        对比训练过程
        """
        print(f"\n{'='*50}")
        print(f"开始严格对拍测试: {test_name}")
        print(f"序列数量: {len(token_sequences)}")
        print(f"总token数: {sum(len(seq) for seq in token_sequences)}")
        print(f"目标merge数: {num_merges}, 最小频次: {min_frequency}")
        print(f"{'='*50}")
        
        # 1. 使用minbpe参考实现
        print("\n🔍 训练minbpe参考实现...")
        minbpe_tokenizer = IntBasicTokenizer()
        minbpe_stats = minbpe_tokenizer.train(
            token_sequences, num_merges, min_frequency, verbose=True
        )
        minbpe_rules = minbpe_tokenizer.get_merge_rules()
        
        # 2. 使用我们的numba实现
        print("\n🔍 训练numba实现...")
        our_engine = BPEEngine(train_backend="numba", encode_backend="python")
        our_stats = our_engine.train(token_sequences, num_merges=num_merges, min_frequency=min_frequency)
        our_rules = our_engine.merge_rules
        
        # 3. 详细对比训练过程
        comparison_result = self._compare_training_details(
            minbpe_stats, minbpe_rules, our_stats, our_rules, test_name
        )
        
        # 4. 对比编码结果
        encode_comparison = self._compare_encoding(
            token_sequences, minbpe_tokenizer, our_engine, test_name
        )
        
        # 合并结果
        final_result = {
            "test_name": test_name,
            "data_info": {
                "num_sequences": len(token_sequences),
                "total_tokens": sum(len(seq) for seq in token_sequences),
                "unique_tokens": len(set(token for seq in token_sequences for token in seq))
            },
            "training_comparison": comparison_result,
            "encoding_comparison": encode_comparison,
            "overall_success": comparison_result["success"] and encode_comparison["success"]
        }
        
        self.results[test_name] = final_result
        return final_result
    
    def _compare_training_details(self, 
                                 minbpe_stats: Dict, 
                                 minbpe_rules: List[Tuple[int, int, int]],
                                 our_stats: Dict, 
                                 our_rules: List[Tuple[int, int, int]],
                                 test_name: str) -> Dict[str, Any]:
        """详细对比训练过程"""
        
        print(f"\n📊 对比训练统计信息:")
        print(f"  minbpe: {minbpe_stats['num_merges_performed']} merges")
        print(f"  我们的: {our_stats['num_merges_performed']} merges")
        
        # 对比merge数量
        if minbpe_stats['num_merges_performed'] != our_stats['num_merges_performed']:
            print(f"❌ Merge数量不一致!")
            return {"success": False, "error": "merge count mismatch"}
        
        # 逐步对比每个merge规则
        print(f"\n📋 逐步对比merge规则:")
        rules_match = True
        rule_details = []
        
        for i, (minbpe_rule, our_rule) in enumerate(zip(minbpe_rules, our_rules)):
            minbpe_pair = (minbpe_rule[0], minbpe_rule[1])
            our_pair = (our_rule[0], our_rule[1])
            
            step_match = (minbpe_pair == our_pair)
            if not step_match:
                rules_match = False
            
            status = "✅" if step_match else "❌"
            print(f"  第{i+1}轮: {status} minbpe:{minbpe_pair} vs 我们的:{our_pair}")
            
            rule_details.append({
                "step": i + 1,
                "minbpe_pair": minbpe_pair,
                "minbpe_new_id": minbpe_rule[2],
                "our_pair": our_pair,
                "our_new_id": our_rule[2],
                "match": step_match
            })
        
        training_log = minbpe_stats.get('training_log', [])
        if training_log:
            print(f"\n📈 训练过程详情:")
            for step_info in training_log:
                print(f"  第{step_info['step']}轮: {step_info['pair']} -> {step_info['new_id']} "
                      f"(频次: {step_info['frequency']}, 剩余tokens: {step_info['remaining_tokens']})")
        
        return {
            "success": rules_match,
            "num_merges": len(minbpe_rules),
            "rule_details": rule_details,
            "minbpe_stats": minbpe_stats,
            "our_stats": our_stats
        }
    
    def _compare_encoding(self, 
                         test_sequences: List[List[int]],
                         minbpe_tokenizer: IntBasicTokenizer,
                         our_engine: BPEEngine,
                         test_name: str) -> Dict[str, Any]:
        """对比编码结果"""
        
        print(f"\n🔧 对比编码结果:")
        
        our_engine.build_encoder()
        encoding_results = []
        all_match = True
        
        # 测试每个序列的编码
        for i, seq in enumerate(test_sequences):
            minbpe_encoded = minbpe_tokenizer.encode(seq)
            our_encoded = our_engine.encode(seq)
            
            match = (minbpe_encoded == our_encoded)
            if not match:
                all_match = False
            
            status = "✅" if match else "❌"
            print(f"  序列{i+1}: {status}")
            if not match:
                print(f"    原始: {seq}")
                print(f"    minbpe: {minbpe_encoded}")
                print(f"    我们的: {our_encoded}")
            
            encoding_results.append({
                "sequence_id": i,
                "original": seq,
                "minbpe_encoded": minbpe_encoded,
                "our_encoded": our_encoded,
                "match": match
            })
        
        return {
            "success": all_match,
            "encoding_results": encoding_results
        }
    
    def save_results(self, filepath: str):
        """保存对比结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 结果已保存到: {filepath}")


def create_synthetic_test_data() -> Dict[str, List[List[int]]]:
    """创建多种合成测试数据"""
    return {
        "simple": [
            [1, 2, 3, 1, 2, 4, 1, 2],
            [1, 2, 5, 3, 4, 1, 2],
            [3, 4, 3, 4, 5, 6],
            [1, 2, 3, 4, 1, 2, 3, 4],
        ],
        
        "complex": [
            [1, 2, 1, 2, 1, 2, 3, 4, 3, 4],  # 高频(1,2)和(3,4)
            [5, 6, 5, 6, 7, 8, 7, 8],        # 高频(5,6)和(7,8)
            [1, 2, 3, 4, 5, 6, 7, 8],        # 混合序列
            [9, 10, 9, 10, 11, 12],          # 中频pairs
            [1, 2, 5, 6, 9, 10],             # 跨组合
        ],
        
        "edge_cases": [
            [1],                              # 单token
            [1, 2],                          # 双token
            [1, 1, 1, 1],                    # 重复token
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 无重复pairs
        ]
    }


def load_real_dataset_sample(dataset_name: str, max_sequences: int = 100) -> List[List[int]]:
    """加载真实数据集样本"""
    try:
        if dataset_name == "qm9test":
            from config import ProjectConfig
            from data.unified_data_interface import UnifiedDataInterface
            
            # 创建配置
            config = ProjectConfig()
            
            # 创建数据接口
            data_interface = UnifiedDataInterface(
                config=config,
                dataset="qm9test"
            )
            
            # 获取序列化数据（使用正确的接口）
            train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = data_interface.get_sequences_by_splits("dfs")
            
            # 使用测试集数据
            token_sequences = []
            for graph_id, sequence in test_seqs[:max_sequences]:
                token_sequences.append(sequence)
            
            return token_sequences[:max_sequences]
            
    except Exception as e:
        print(f"❌ 加载{dataset_name}失败: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    return []


def main():
    """主测试函数"""
    print("🚀 开始严格BPE对拍测试")
    
    comparator = StrictBPEComparison()
    
    # 1. 合成数据测试
    print("\n" + "="*60)
    print("第一阶段: 合成数据测试")
    print("="*60)
    
    synthetic_data = create_synthetic_test_data()
    
    for test_name, sequences in synthetic_data.items():
        try:
            result = comparator.compare_training(
                sequences, 
                num_merges=5, 
                min_frequency=2,
                test_name=f"synthetic_{test_name}"
            )
            
            status = "✅ 通过" if result["overall_success"] else "❌ 失败"
            print(f"\n📊 {test_name} 测试结果: {status}")
            
        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. 真实数据集测试（样本）
    print("\n" + "="*60)
    print("第二阶段: 真实数据集测试")
    print("="*60)
    
    real_datasets = ["qm9test"]  # 可以添加更多数据集
    
    for dataset_name in real_datasets:
        try:
            print(f"\n🔍 加载 {dataset_name} 数据...")
            sequences = load_real_dataset_sample(dataset_name, max_sequences=20)
            
            if not sequences:
                print(f"⚠️ {dataset_name} 数据为空，跳过")
                continue
                
            result = comparator.compare_training(
                sequences,
                num_merges=10,
                min_frequency=2,
                test_name=f"real_{dataset_name}"
            )
            
            status = "✅ 通过" if result["overall_success"] else "❌ 失败"
            print(f"\n📊 {dataset_name} 测试结果: {status}")
            
        except Exception as e:
            print(f"❌ {dataset_name} 测试出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 保存结果
    comparator.save_results("bpe_comparison_results.json")
    
    # 4. 总结报告
    print("\n" + "="*60)
    print("测试总结报告")
    print("="*60)
    
    total_tests = len(comparator.results)
    passed_tests = sum(1 for r in comparator.results.values() if r["overall_success"])
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！实现与minbpe严格一致！")
        return True
    else:
        print("\n❌ 存在失败测试，需要进一步检查！")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
