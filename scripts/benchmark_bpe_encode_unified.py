#!/usr/bin/env python3
"""BPE编码性能测试（BPEEngine统一接口，多模式）"""
#!/usr/bin/env python3
"""
BPE encode 统一基准测试脚本（基于 BPEEngine）
================================================

目标：
- 使用统一的 BPEEngine 进行BPE编码基准测试
- 支持多种后端：cpp/numba/python
- 支持多种编码模式：all/topk/random/gaussian
- 从 UDI 读取真实数据或使用随机序列
- 关注小 batch 在线延迟与大 batch 吞吐

用法示例：
  # 真实数据，对比不同后端
  python scripts/benchmark_bpe_encode_unified.py \
    --dataset qm9test --version latest --method eulerian --source raw \
    --backends cpp numba python --mode latency --batch-size 64 --repeat 100

  # 随机数据，测试不同编码模式
  python scripts/benchmark_bpe_encode_unified.py \
    --random 1 --num-seqs 10000 --seq-len 20 \
    --backends cpp --modes all topk random \
    --mode throughput --batch-size 512 --repeat 30 --save result.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import random
import numpy as np

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.compression.bpe_engine import BPEEngine


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    backend: str
    encode_mode: str
    batch_size: int
    num_sequences: int
    avg_time_ms: float
    throughput_seq_per_sec: float
    memory_mb: float
    percentiles: Dict[str, float]


def get_memory_mb() -> float:
    """获取当前内存使用量（MB）"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def load_sequences_via_udi(cfg: ProjectConfig, method: str, source: str, version: str) -> List[List[int]]:
    """通过UDI加载序列数据"""
    print(f"📂 从UDI加载 {source} 序列: dataset={cfg.dataset.name}, method={method}")
    
    udi = UnifiedDataInterface(cfg, cfg.dataset.name)
    if source == "raw":
        seqs_with_ids, _ = udi.get_sequences(method)
        seqs = [seq for _, seq in seqs_with_ids]
    else:
        raise ValueError(f"不支持的源类型: {source}")
    
    print(f"✅ 加载完成: {len(seqs)} 个序列")
    return seqs


def load_bpe_codebook_via_udi(cfg: ProjectConfig, method: str, version: str) -> Dict[str, Any]:
    """通过UDI加载BPE codebook"""
    print(f"📂 从UDI加载 BPE codebook: method={method}")
    
    udi = UnifiedDataInterface(cfg, cfg.dataset.name)
    codebook = udi.get_bpe_codebook(method)
    
    print(f"✅ BPE codebook加载完成: {len(codebook['merge_rules'])} merge rules, vocab_size={codebook['vocab_size']}")
    return codebook


def generate_random_sequences(num_seqs: int, seq_len: int, vocab_size: int = 50) -> List[List[int]]:
    """生成随机序列用于测试"""
    print(f"🎲 生成随机序列: {num_seqs} 个序列, 长度={seq_len}, vocab_size={vocab_size}")
    
    sequences = []
    for _ in range(num_seqs):
        length = random.randint(max(1, seq_len - 5), seq_len + 5)  # 长度有一定随机性
        seq = [random.randint(0, vocab_size - 1) for _ in range(length)]
        sequences.append(seq)
    
    return sequences


def create_bpe_engine(codebook: Dict[str, Any], backend: str, encode_mode: str, **kwargs) -> BPEEngine:
    """创建配置好的BPE引擎"""
    engine = BPEEngine(
        train_backend="python",  # 不需要训练
        encode_backend=backend,
        encode_rank_mode=encode_mode,
        **kwargs
    )
    
    # 设置训练好的参数
    engine.merge_rules = [tuple(map(int, rule)) for rule in codebook['merge_rules']]
    engine.vocab_size = int(codebook['vocab_size'])
    engine.build_encoder()
    
    return engine


def take_batch(seqs: List[List[int]], batch_size: int, offset: int) -> List[List[int]]:
    """获取批次数据"""
    n = len(seqs)
    if n == 0:
        return []
    start = (offset * batch_size) % n
    end = start + batch_size
    if end <= n:
        return seqs[start:end]
    return seqs[start:] + seqs[: end - n]


def run_latency_benchmark(
    engine: BPEEngine,
    sequences: List[List[int]],
    batch_size: int,
    repeat: int
) -> BenchmarkResult:
    """运行延迟基准测试"""
    backend = engine.encode_backend
    encode_mode = engine.encode_rank_mode
    
    print(f"🔬 延迟测试: backend={backend}, mode={encode_mode}, batch_size={batch_size}, repeat={repeat}")
    
    # 预热
    warmup_batch = take_batch(sequences, min(batch_size, 10), 0)
    for _ in range(3):
        _ = engine.batch_encode(warmup_batch)
    
    # 基准测试
    times = []
    memory_before = get_memory_mb()
    
    for i in range(repeat):
        batch = take_batch(sequences, batch_size, i)
        
        start_time = time.perf_counter()
        _ = engine.batch_encode(batch)
        end_time = time.perf_counter()
        
        times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    memory_after = get_memory_mb()
    
    # 统计
    avg_time_ms = np.mean(times)
    throughput = (batch_size * repeat) / (sum(times) / 1000)  # seq/sec
    percentiles = {
        'p50': np.percentile(times, 50),
        'p90': np.percentile(times, 90),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99)
    }
    
    return BenchmarkResult(
        backend=backend,
        encode_mode=encode_mode,
        batch_size=batch_size,
        num_sequences=batch_size * repeat,
        avg_time_ms=avg_time_ms,
        throughput_seq_per_sec=throughput,
        memory_mb=memory_after - memory_before,
        percentiles=percentiles
    )


def run_throughput_benchmark(
    engine: BPEEngine,
    sequences: List[List[int]],
    batch_size: int,
    repeat: int
) -> BenchmarkResult:
    """运行吞吐量基准测试"""
    backend = engine.encode_backend
    encode_mode = engine.encode_rank_mode
    
    print(f"🚀 吞吐量测试: backend={backend}, mode={encode_mode}, batch_size={batch_size}, repeat={repeat}")
    
    # 预热
    warmup_batch = take_batch(sequences, min(batch_size, 100), 0)
    for _ in range(5):
        _ = engine.batch_encode(warmup_batch)
    
    # 基准测试
    total_sequences = 0
    total_time = 0
    memory_before = get_memory_mb()
    
    for i in range(repeat):
        batch = take_batch(sequences, batch_size, i)
        
        start_time = time.perf_counter()
        _ = engine.batch_encode(batch)
        end_time = time.perf_counter()
        
        total_sequences += len(batch)
        total_time += (end_time - start_time)
    
    memory_after = get_memory_mb()
    
    # 统计
    avg_time_ms = (total_time / repeat) * 1000  # 平均每批次时间（毫秒）
    throughput = total_sequences / total_time  # seq/sec
    
    return BenchmarkResult(
        backend=backend,
        encode_mode=encode_mode,
        batch_size=batch_size,
        num_sequences=total_sequences,
        avg_time_ms=avg_time_ms,
        throughput_seq_per_sec=throughput,
        memory_mb=memory_after - memory_before,
        percentiles={}  # 吞吐量测试不计算百分位
    )


def print_results(results: List[BenchmarkResult], mode: str):
    """打印基准测试结果"""
    print(f"\n📊 基准测试结果 ({mode} 模式)")
    print("=" * 80)
    
    if mode == "latency":
        print(f"{'Backend':<10} {'Mode':<10} {'BatchSize':<10} {'AvgTime(ms)':<12} {'Throughput':<12} {'P50':<8} {'P90':<8} {'P95':<8} {'P99':<8}")
        print("-" * 80)
        for result in results:
            print(f"{result.backend:<10} {result.encode_mode:<10} {result.batch_size:<10} "
                  f"{result.avg_time_ms:<12.3f} {result.throughput_seq_per_sec:<12.1f} "
                  f"{result.percentiles['p50']:<8.2f} {result.percentiles['p90']:<8.2f} "
                  f"{result.percentiles['p95']:<8.2f} {result.percentiles['p99']:<8.2f}")
    else:
        print(f"{'Backend':<10} {'Mode':<10} {'BatchSize':<10} {'Sequences':<10} {'Throughput':<12} {'Memory(MB)':<10}")
        print("-" * 80)
        for result in results:
            print(f"{result.backend:<10} {result.encode_mode:<10} {result.batch_size:<10} "
                  f"{result.num_sequences:<10} {result.throughput_seq_per_sec:<12.1f} {result.memory_mb:<10.2f}")


def save_results(results: List[BenchmarkResult], filepath: str):
    """保存结果到JSON文件"""
    data = []
    for result in results:
        data.append({
            'backend': result.backend,
            'encode_mode': result.encode_mode,
            'batch_size': result.batch_size,
            'num_sequences': result.num_sequences,
            'avg_time_ms': result.avg_time_ms,
            'throughput_seq_per_sec': result.throughput_seq_per_sec,
            'memory_mb': result.memory_mb,
            'percentiles': result.percentiles
        })
    
    with open(filepath, 'w') as f:
        json.dump({
            'results': data,
            'metadata': {
                'script': 'benchmark_bpe_encode_unified.py',
                'timestamp': time.time()
            }
        }, f, indent=2)
    
    print(f"💾 结果已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="BPE encode 统一基准测试")
    
    # 数据源
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset", help="数据集名称 (如 qm9test)")
    data_group.add_argument("--random", type=int, help="使用随机数据 (设为1启用)")
    
    # UDI数据参数
    parser.add_argument("--version", default="latest", help="数据版本")
    parser.add_argument("--method", default="eulerian", help="序列化方法")
    parser.add_argument("--source", default="raw", choices=["raw"], help="序列来源")
    
    # 随机数据参数
    parser.add_argument("--num-seqs", type=int, default=10000, help="随机序列数量")
    parser.add_argument("--seq-len", type=int, default=20, help="随机序列长度")
    parser.add_argument("--vocab-size", type=int, default=50, help="随机序列词表大小")
    
    # 基准测试参数
    parser.add_argument("--backends", nargs="+", default=["cpp"], 
                       choices=["cpp", "numba", "python"], help="编码后端")
    parser.add_argument("--modes", nargs="+", default=["all"],
                       choices=["all", "topk", "random", "gaussian"], help="编码模式")
    parser.add_argument("--mode", choices=["latency", "throughput"], default="latency", help="测试模式")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--repeat", type=int, default=100, help="重复次数")
    
    # 输出参数
    parser.add_argument("--save", help="保存结果到JSON文件")
    parser.add_argument("--config", help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.config:
        config = ProjectConfig(config_path=args.config)
    else:
        config = ProjectConfig()
    
    # 设置环境
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    # 加载数据
    if args.dataset:
        config.dataset.name = args.dataset
        sequences = load_sequences_via_udi(config, args.method, args.source, args.version)
        codebook = load_bpe_codebook_via_udi(config, args.method, args.version)
    else:
        sequences = generate_random_sequences(args.num_seqs, args.seq_len, args.vocab_size)
        # 创建虚拟codebook用于测试
        codebook = {
            'merge_rules': [(i, i+1, 100+i) for i in range(0, min(20, args.vocab_size-2), 2)],
            'vocab_size': args.vocab_size + 10
        }
    
    print(f"📊 序列统计: {len(sequences)} 个序列, 平均长度: {np.mean([len(s) for s in sequences]):.1f}")
    
    # 运行基准测试
    results = []
    
    for backend in args.backends:
        for encode_mode in args.modes:
            try:
                # 为不同模式设置参数
                kwargs = {}
                if encode_mode == "topk":
                    kwargs["encode_rank_k"] = 100
                elif encode_mode in ["random", "gaussian"]:
                    kwargs["encode_rank_min"] = 0
                    kwargs["encode_rank_max"] = len(codebook['merge_rules'])
                
                # 创建引擎
                engine = create_bpe_engine(codebook, backend, encode_mode, **kwargs)
                
                # 运行基准测试
                if args.mode == "latency":
                    result = run_latency_benchmark(engine, sequences, args.batch_size, args.repeat)
                else:
                    result = run_throughput_benchmark(engine, sequences, args.batch_size, args.repeat)
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ 测试失败: backend={backend}, mode={encode_mode}, error={e}")
                continue
    
    # 输出结果
    if results:
        print_results(results, args.mode)
        
        if args.save:
            save_results(results, args.save)
    else:
        print("❌ 没有成功的测试结果")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

