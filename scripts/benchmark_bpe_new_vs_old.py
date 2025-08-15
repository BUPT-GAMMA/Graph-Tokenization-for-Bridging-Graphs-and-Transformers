#!/usr/bin/env python3
"""
BPE 训练性能对比脚本（基线 vs 新引擎）
- 基线：standard 或 minbpe
- 新引擎：BPEEngine（训练后端固定 cpp）
- 输出：屏幕打印与 JSON 结果落盘
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.compression.bpe_engine import BPEEngine
from src.algorithms.compression.main_bpe import StandardBPECompressor
from foreign_dataset_files_to_convert.int_basic_tokenizer import IntBasicTokenizer

def load_test_data(dataset: str = "qm9test", method: str = "feuler", limit: int = None) -> List[List[int]]:
    """加载测试数据"""
    print(f"📂 加载 {dataset} 数据集，序列化方法：{method}")
    
    config = ProjectConfig()
    config.dataset.name = dataset
    
    udi = UnifiedDataInterface(config=config, dataset=dataset)
    sequences_with_ids, _ = udi.get_sequences(method)
    sequences = [seq for _, seq in sequences_with_ids]
    
    if limit and len(sequences) > limit:
        sequences = sequences[:limit]
    
    # 确保所有元素都是int类型
    sequences = [[int(token) for token in seq] for seq in sequences]
    
    print("✅ 数据加载完成：")
    print(f"  序列数量: {len(sequences)}")
    total_tokens = sum(len(seq) for seq in sequences)
    print(f"  总token数: {total_tokens}")
    print(f"  平均序列长度: {total_tokens / len(sequences):.1f}")
    
    return sequences

def benchmark_standard_bpe(sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict[str, Any]:
    """测试标准BPE实现"""
    print("🔧 测试标准BPE实现...")
    
    compressor = StandardBPECompressor(
        num_merges=num_merges, 
        min_frequency=min_frequency, 
        debug=False
    )
    
    start_time = time.perf_counter()
    stats = compressor.train(sequences)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    
    return {
        "method": "standard_bpe",
        "elapsed_time": elapsed,
        "num_merges_performed": stats.get("num_merges_performed", len(compressor.merge_rules)),
        "final_vocab_size": len(compressor.token_to_id)
    }

def benchmark_minbpe_reference(sequences: List[List[int]], num_merges: int, min_frequency: int) -> Dict[str, Any]:
    """minBPE 参考实现（用于正确性/语义基线）"""
    print("🔍 测试 minBPE 参考实现...")
    tok = IntBasicTokenizer()
    start_time = time.perf_counter()
    stats = tok.train(sequences, num_merges=num_merges, min_frequency=min_frequency, verbose=False)
    end_time = time.perf_counter()
    return {
        "method": "minbpe_ref",
        "elapsed_time": end_time - start_time,
        "num_merges_performed": stats.get("num_merges_performed", len(tok.merges)),
        "final_vocab_size": len(tok.vocab)
    }

def benchmark_engine(sequences: List[List[int]], num_merges: int, min_frequency: int, train_backend: str = "cpp") -> Dict[str, Any]:
    """测试新版本BPE Engine（训练后端仅支持 cpp）。"""
    tag = "cpp"
    print(f"🚀 测试新版本BPE Engine（{tag}）...")
    
    engine = BPEEngine(train_backend=train_backend, encode_backend="python")
    
    start_time = time.perf_counter()
    stats = engine.train(sequences, num_merges=num_merges, min_frequency=min_frequency)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    
    return {
        "method": f"new_engine_{tag}",
        "elapsed_time": elapsed,
        "num_merges_performed": stats["num_merges_performed"],
        "final_vocab_size": stats["final_vocab_size"]
    }

def run_benchmark(dataset: str = "qm9test", num_merges: int = 100, min_frequency: int = 10, limit: int = None, *, baseline: str = "standard", engine_backend: str = "numba"):
    """运行完整的性能对比测试"""
    
    print("=" * 80)
    print("🎯 BPE训练性能对比测试")
    print("=" * 80)
    print(f"数据集: {dataset}")
    print(f"merge数量: {num_merges}")
    print(f"最小频次: {min_frequency}")
    if limit:
        print(f"序列限制: {limit}")
    print()
    
    # 加载数据
    sequences = load_test_data(dataset, limit=limit)
    print()
    
    # 运行测试
    results = {}
    
    try:
        # 1. 基线
        results[baseline] = (benchmark_standard_bpe if baseline == "standard" else benchmark_minbpe_reference)(
            sequences, num_merges, min_frequency
        )
        print(f"✅ 基线({baseline})完成：{results[baseline]['elapsed_time']:.2f}秒")
        print()
        # 2. 新版本Engine
        results["new_engine"] = benchmark_engine(sequences, num_merges, min_frequency, train_backend="cpp")
        print(f"✅ 新版本Engine完成：{results['new_engine']['elapsed_time']:.2f}秒")
        print()
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 性能对比分析
    print("=" * 80)
    print("📊 性能对比结果")
    print("=" * 80)
    
    base_time = results[baseline]["elapsed_time"]
    new_time = results["new_engine"]["elapsed_time"]
    
    print(f"基线({baseline})：      {base_time:.3f}秒")
    print(f"新版本({engine_backend})：     {new_time:.3f}秒")
    print()
    
    # 计算加速比
    speedup = base_time / new_time if new_time > 0 else float('inf')
    
    print("🚀 加速比：")
    print(f"  {engine_backend} vs {baseline}：  {speedup:.2f}x")
    print()
    
    # 验证结果一致性
    print("🔍 结果验证：")
    std_merges = results[baseline]["num_merges_performed"]
    new_merges = results["new_engine"]["num_merges_performed"]
    
    print(f"  基线 merge数：     {std_merges}")
    print(f"  新版本({engine_backend}) merge数：    {new_merges}")
    
    if std_merges == new_merges:
        print("✅ merge数量一致")
    else:
        print("❌ merge数量不一致！")
    
    # 保存结果
    output_file = f"benchmark_results_{dataset}_{num_merges}merges.json"
    detailed_results = {
        "test_config": {
            "dataset": dataset,
            "num_merges": num_merges,
            "min_frequency": min_frequency,
            "sequence_limit": limit,
            "num_sequences": len(sequences),
            "total_tokens": sum(len(seq) for seq in sequences)
        },
        "results": results,
        "performance_summary": {
            "speedup": speedup,
            "fastest_method": f"new_engine_{engine_backend}" if new_time < base_time else baseline
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"📝 详细结果已保存至：{output_file}")
    
    return detailed_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BPE训练性能对比测试")
    parser.add_argument("--dataset", default="qm9test", choices=["qm9test", "qm9", "zinc", "aqsol"],
                       help="测试数据集")
    parser.add_argument("--num-merges", type=int, default=2000, help="merge操作数量")
    parser.add_argument("--min-frequency", type=int, default=100, help="最小频次阈值")
    parser.add_argument("--limit", type=int, help="限制序列数量（用于快速测试）")
    parser.add_argument("--baseline", choices=["standard", "minbpe"], default="standard", help="对比基线：standard 或 minbpe")
    # 训练后端固定为 cpp；保留参数以兼容但忽略其值
    parser.add_argument("--engine-backend", choices=["cpp"], default="cpp", help="新版本引擎训练后端：仅 cpp")
    
    args = parser.parse_args()
    
    run_benchmark(
        dataset=args.dataset,
        num_merges=args.num_merges,
        min_frequency=args.min_frequency,
        limit=args.limit,
        baseline=args.baseline,
        engine_backend=args.engine_backend
    )