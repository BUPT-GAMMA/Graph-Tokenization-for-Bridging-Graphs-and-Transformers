#!/usr/bin/env python3
"""
效率指标收集脚本
================

目标：
1. 逐个测试每个数据集的每个序列化方法
2. 收集序列化速度、长度、BPE压缩效果等指标
3. 支持多重采样配置，不同采样次数的结果保存到不同文件
4. 保存为CSV格式供exp1_speed的画图脚本使用
5. 不覆盖现有的序列化数据，仅做测试和指标收集

输出文件：
- exp1_speed/token_length/{dataset}_mult{N}_token_length.csv (包含多重采样信息)
- exp1_speed/serialize_time/{dataset}_serialize_speed.csv (序列化速度，不受多重采样影响)
- (可选)其他相关指标文件

多重采样说明：
- 多重采样次数>1时启用，会为每个图生成多个序列变体
- 序列化速度不受多重采样影响（因为是相同的图处理）
- token长度会随多重采样次数线性增加
- 不同多重采样次数的结果保存到不同CSV文件中

使用方法：
python final/collect_efficiency_metrics.py --datasets zinc,colors3 --methods dfs,bfs,smiles --mult 5
python final/collect_efficiency_metrics.py --datasets qm9test --mult 1 3 5  # 比较不同多重采样次数
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import csv
import traceback

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.algorithms.compression.main_bpe import StandardBPECompressor


def collect_single_method_metrics(dataset: str, method: str,
                                 bpe_num_merges: int = 2000,
                                 bpe_min_frequency: int = 2,
                                 limit: Optional[int] = None,
                                 mult: Optional[int] = None,
                                 udi: Optional[UnifiedDataInterface] = None,
                                 loader: Optional[Any] = None,
                                 graphs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    收集单个方法的全部性能指标
    
    Args:
        dataset: 数据集名称
        method: 序列化方法名称
        bpe_num_merges: BPE合并次数
        bpe_min_frequency: BPE最小频率
        limit: 限制处理的图数量（用于快速测试）
        
    Returns:
        包含所有指标的字典
    """
    print(f"🔍 开始测试: {dataset} - {method}")
    
    try:
        # 1. 复用（或构建）数据接口/加载器/图数据
        if udi is None or loader is None or graphs is None:
            config = ProjectConfig()
            config.dataset.name = dataset
            # 多重采样配置（与数据预处理保持一致语义）
            if mult is not None and int(mult) > 1:
                # 打开多重采样
                setattr(config.serialization.multiple_sampling, 'enabled', True)
                setattr(config.serialization.multiple_sampling, 'num_realizations', int(mult))
            else:
                setattr(config.serialization.multiple_sampling, 'enabled', False)
                setattr(config.serialization.multiple_sampling, 'num_realizations', 1)
            udi = UnifiedDataInterface(config=config, dataset=dataset)
            udi.preload_graphs()
            loader = udi.get_dataset_loader()
            graphs = udi.get_graphs()
            if limit is not None:
                graphs = graphs[:limit]
        # 信息打印
        if limit is not None:
            print(f"   限制处理图数量: {len(graphs)}")
        print(f"   数据集大小: {len(graphs)} 个图")
        
        # 2. 序列化测试
        print(f"   🚀 开始序列化测试...")
        serializer = SerializerFactory.create_serializer(method)
        
        # 设置适当的并发度（避免过度占用资源）
        # serializer.stats_num_workers = max(1, min(4, os.cpu_count() // 2))
        serializer.initialize_with_dataset(loader, graphs)
        
        # 测量序列化时间
        serialization_start = time.perf_counter()
        if mult is not None and int(mult) > 1:
            batch_results = serializer.batch_multiple_serialize(
                graphs,
                num_samples=int(mult),
                desc=f"serialize-multi-{method}",
                parallel=True,
            )
        else:
            batch_results = serializer.batch_serialize(graphs, desc=f"serialize-{method}")
        serialization_end = time.perf_counter()
        
        # 3. 提取序列化结果和统计
        sequences: List[List[int]] = []
        for res in batch_results:
            if res and getattr(res, 'token_sequences', None):
                if mult is not None and int(mult) > 1:
                    # 展开多重采样的变体
                    for seq in res.token_sequences:
                        sequences.append(seq)
                else:
                    sequences.append(res.token_sequences[0])
            else:
                sequences.append([])
        
        # 过滤空序列
        sequences_nonempty = [s for s in sequences if len(s) > 0]
        
        if not sequences_nonempty:
            raise ValueError(f"序列化方法 {method} 没有产生有效序列")
        
        # 序列化指标计算
        serialization_time = max(serialization_end - serialization_start, 1e-9)
        total_graphs = len(graphs)
        serialization_speed = total_graphs / serialization_time  # graphs/second
        
        # 序列长度统计
        total_tokens = sum(len(s) for s in sequences_nonempty)
        avg_sequence_length = total_tokens / len(sequences_nonempty)
        
        if mult is not None and int(mult) > 1:
            print(f"   ✅ 序列化完成(多重采样x{int(mult)}): {serialization_speed:.1f} graphs/sec, 平均长度: {avg_sequence_length:.1f}")
        else:
            print(f"   ✅ 序列化完成: {serialization_speed:.1f} graphs/sec, 平均长度: {avg_sequence_length:.1f}")
        
        # 4. BPE压缩测试
        print(f"   🎓 开始BPE训练...")
        compressor = StandardBPECompressor(
            num_merges=bpe_num_merges, 
            min_frequency=bpe_min_frequency,
            debug=False
        )
        
        # 训练BPE
        bpe_train_start = time.perf_counter()
        bpe_train_stats = compressor.train(sequences_nonempty)
        bpe_train_end = time.perf_counter()
        
        bpe_training_time = max(bpe_train_end - bpe_train_start, 1e-9)
        
        # 5. BPE压缩测试
        print(f"   📦 开始BPE压缩...")
        bpe_encode_start = time.perf_counter()
        encoded_sequences = [compressor.encode(seq) for seq in sequences_nonempty]
        bpe_encode_end = time.perf_counter()
        
        bpe_encoding_time = max(bpe_encode_end - bpe_encode_start, 1e-9)
        
        # 6. 压缩效果统计
        compressed_total_tokens = sum(len(s) for s in encoded_sequences)
        compression_rate = compressed_total_tokens / total_tokens if total_tokens > 0 else 1.0
        compression_ratio = 1.0 / compression_rate if compression_rate > 0 else 1.0
        tokens_saved = total_tokens - compressed_total_tokens
        
        avg_compressed_length = compressed_total_tokens / len(encoded_sequences)
        
        print(f"   ✅ BPE完成: 压缩率 {compression_rate:.3f}, 压缩比 {compression_ratio:.1f}x")
        
        # 7. 质量验证（少量样本）
        sample_size = min(100, len(sequences_nonempty))
        correct_decodes = 0
        for i in range(sample_size):
            try:
                decoded = compressor.decode(encoded_sequences[i])
                if decoded == sequences_nonempty[i]:
                    correct_decodes += 1
            except:
                pass
        
        decode_accuracy = correct_decodes / sample_size if sample_size > 0 else 0.0
        
        # 8. 汇总所有指标
        metrics = {
            'dataset': dataset,
            'method': method,
            'success': True,
            # 多重采样配置回写
            'multiple_sampling_enabled': (int(mult) > 1) if mult is not None else False,
            'multiple_sampling_num_realizations': int(mult) if (mult is not None and int(mult) > 0) else 1,
            
            # 基础统计
            'total_graphs': total_graphs,
            'valid_sequences': len(sequences_nonempty),
            
            # 序列化指标
            'serialization_time_seconds': serialization_time,
            'serialization_speed_graphs_per_sec': serialization_speed,
            
            # 序列长度指标
            'original_total_tokens': total_tokens,
            'original_avg_length': avg_sequence_length,
            
            # BPE训练指标
            'bpe_training_time_seconds': bpe_training_time,
            'bpe_encoding_time_seconds': bpe_encoding_time,
            'bpe_vocab_size': bpe_train_stats.get('final_vocab_size', 0),
            'bpe_num_merges_performed': bpe_train_stats.get('num_merges_performed', 0),
            
            # BPE压缩指标
            'compressed_total_tokens': compressed_total_tokens,
            'compressed_avg_length': avg_compressed_length,
            'compression_rate': compression_rate,  # 压缩后/压缩前
            'compression_ratio': compression_ratio,  # 压缩前/压缩后
            'tokens_saved': tokens_saved,
            'compression_percentage': (1.0 - compression_rate) * 100.0,
            
            # 质量指标
            'decode_accuracy': decode_accuracy,
            'decode_sample_size': sample_size,
            
            # 时间戳
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            
            # 配置信息
            'bpe_num_merges': bpe_num_merges,
            'bpe_min_frequency': bpe_min_frequency,
            'limit': limit
        }
        
        return metrics
        
    except Exception as e:
        print(f"   ❌ 测试失败: {str(e)}")
        return {
            'dataset': dataset,
            'method': method,
            'success': False,
            'error': str(e),
            'error_traceback': traceback.format_exc(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }


def save_metrics_to_csv(all_metrics: List[Dict[str, Any]], output_dir: Path):
    """
    将收集的指标保存为CSV文件，按照exp1_speed画图脚本的格式要求
    """
    print("💾 保存指标到CSV文件...")
    
    # 过滤成功的结果
    successful_metrics = [m for m in all_metrics if m.get('success', False)]
    
    if not successful_metrics:
        print("❌ 没有成功的测试结果")
        return
    
    # 按数据集分组
    datasets = set(m['dataset'] for m in successful_metrics)
    
    for dataset in datasets:
        dataset_metrics = [m for m in successful_metrics if m['dataset'] == dataset]
        
        # 1. 保存序列化长度对比数据 (token_length)
        token_length_dir = output_dir / "token_length"
        token_length_dir.mkdir(parents=True, exist_ok=True)

        # 获取多重采样信息（使用第一个成功的结果）
        mult_info = dataset_metrics[0] if dataset_metrics else {}
        mult_enabled = mult_info.get('multiple_sampling_enabled', False)
        mult_num = mult_info.get('multiple_sampling_num_realizations', 1)

        # 构建文件名，包含多重采样信息
        mult_suffix = f"_mult{mult_num}" if mult_enabled and mult_num > 1 else "_mult1"
        token_length_file = token_length_dir / f"{dataset}{mult_suffix}_token_length.csv"

        with open(token_length_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 不做单位转换与四舍五入，直接输出原始值
            writer.writerow(['serialization_method', 'original_total_tokens', 'compressed_total_tokens', 'compression_ratio'])
            for m in dataset_metrics:
                compression_ratio = (m['original_total_tokens'] / m['compressed_total_tokens']) if m['compressed_total_tokens'] > 0 else 1.0
                writer.writerow([
                    m['method'],
                    m['original_total_tokens'],
                    m['compressed_total_tokens'],
                    compression_ratio
                ])

        print(f"✅ 序列化长度数据: {token_length_file}")
        
        # 2. 保存序列化速度对比数据 (serialize_time)
        serialize_time_dir = output_dir / "serialize_time"
        serialize_time_dir.mkdir(parents=True, exist_ok=True)
        
        serialize_time_file = serialize_time_dir / f"{dataset}_serialize_speed.csv"
        with open(serialize_time_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['serialization_method', 'graphs_per_second'])
            for m in dataset_metrics:
                writer.writerow([
                    m['method'],
                    m['serialization_speed_graphs_per_sec']
                ])
        
        print(f"✅ 序列化速度数据: {serialize_time_file}")
        
        # 3. 保存详细指标 (用于进一步分析)
        detailed_dir = output_dir / "detailed_metrics"
        detailed_dir.mkdir(parents=True, exist_ok=True)
        
        detailed_file = detailed_dir / f"{dataset}_detailed_metrics.csv"
        with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
            if dataset_metrics:
                # 使用第一个成功的结果来确定列名
                fieldnames = list(dataset_metrics[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dataset_metrics)
        
        print(f"✅ 详细指标数据: {detailed_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="效率指标收集脚本")
    parser.add_argument("--datasets", type=str, default="qm9test", 
                       help="逗号分隔的数据集列表，默认: qm9test")
    parser.add_argument("--methods", type=str, default=None,
                       help="逗号分隔的序列化方法列表，默认使用所有可用方法")
    parser.add_argument("--bpe_merges", type=int, default=2000,
                       help="BPE合并次数，默认: 2000")
    parser.add_argument("--bpe_min_freq", type=int, default=2,
                       help="BPE最小频率，默认: 2")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制每个数据集处理的图数量（用于快速测试）")
    parser.add_argument("--mult", type=int, default=None,
                       help="多重采样次数，>1 时启用多重采样")
    parser.add_argument("--output", type=str, default=None,
                       help="输出目录，默认为当前脚本所在目录")
    parser.add_argument("--from_detailed", action="store_true",
                       help="从 detailed_metrics 生成 CSV（跳过重新评测）")
    
    args = parser.parse_args()
    
    # 解析数据集列表
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    
    # 解析序列化方法列表
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = SerializerFactory.get_available_serializers()
    
    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent
    
    print("📋 效率指标收集配置:")
    print(f"   数据集: {datasets}")
    print(f"   方法: {methods}")
    print(f"   BPE配置: merges={args.bpe_merges}, min_freq={args.bpe_min_freq}")
    print(f"   限制数量: {args.limit or '无限制'}")
    print(f"   输出目录: {output_dir}")
    print()
    
    # 如果仅从详细指标生成 CSV
    if args.from_detailed:
        print("📄 从 detailed_metrics 导出 CSV ...")
        # 聚合每个数据集的详细指标文件
        for dataset in datasets:
            detailed_file = output_dir / "detailed_metrics" / f"{dataset}_detailed_metrics.csv"
            if not detailed_file.exists():
                print(f"   ⚠️ 跳过: 未找到 {detailed_file}")
                continue
            with open(detailed_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print(f"   ⚠️ 跳过: 文件为空 {detailed_file}")
                continue
            # 获取多重采样信息（从第一个成功的行中提取）
            mult_info = None
            for m in rows:
                if m.get('success') in (True, 'True', 'true', '1', 1):
                    mult_info = m
                    break

            if mult_info:
                mult_enabled = mult_info.get('multiple_sampling_enabled', 'False').lower() in ('true', '1')
                mult_num = int(mult_info.get('multiple_sampling_num_realizations', '1'))
            else:
                mult_enabled = False
                mult_num = 1

            # 构建文件名，包含多重采样信息
            mult_suffix = f"_mult{mult_num}" if mult_enabled and mult_num > 1 else "_mult1"

            # 直接写出两个CSV（不改变单位/不舍入）
            token_length_dir = output_dir / "token_length"
            token_length_dir.mkdir(parents=True, exist_ok=True)
            token_length_file = token_length_dir / f"{dataset}{mult_suffix}_token_length.csv"
            with open(token_length_file, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['serialization_method', 'original_total_tokens', 'compressed_total_tokens', 'compression_ratio'])
                for m in rows:
                    if m.get('success') not in (True, 'True', 'true', '1', 1):
                        continue
                    method = m['method']
                    original_total_tokens = float(m['original_total_tokens'])
                    compressed_total_tokens = float(m['compressed_total_tokens'])
                    compression_ratio = (original_total_tokens / compressed_total_tokens) if float(compressed_total_tokens) > 0 else 1.0
                    w.writerow([method, original_total_tokens, compressed_total_tokens, compression_ratio])

            serialize_time_dir = output_dir / "serialize_time"
            serialize_time_dir.mkdir(parents=True, exist_ok=True)
            serialize_time_file = serialize_time_dir / f"{dataset}_serialize_speed.csv"
            with open(serialize_time_file, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['serialization_method', 'graphs_per_second'])
                for m in rows:
                    if m.get('success') not in (True, 'True', 'true', '1', 1):
                        continue
                    method = m['method']
                    graphs_per_second = float(m['serialization_speed_graphs_per_sec'])
                    w.writerow([method, graphs_per_second])

            print(f"   ✅ 已从详细指标导出: {dataset}")
        print("完成。")
        return

    # 收集所有指标（复用同一UDI/loader/graphs）
    all_metrics = []
    total_tasks = len(datasets) * len(methods)
    current_task = 0
    start_time = time.time()
    for dataset in datasets:
        print(f"📊 处理数据集: {dataset}")
        print("=" * 60)
        # 准备复用资源（将 mult 注入配置）
        config = ProjectConfig()
        config.dataset.name = dataset
        if args.mult is not None and int(args.mult) > 1:
            setattr(config.serialization.multiple_sampling, 'enabled', True)
            setattr(config.serialization.multiple_sampling, 'num_realizations', int(args.mult))
        else:
            setattr(config.serialization.multiple_sampling, 'enabled', False)
            setattr(config.serialization.multiple_sampling, 'num_realizations', 1)
        shared_udi = UnifiedDataInterface(config=config, dataset=dataset)
        shared_udi.preload_graphs()
        shared_loader = shared_udi.get_dataset_loader()
        shared_graphs = shared_udi.get_graphs()

        if args.limit is not None:
            shared_graphs = shared_graphs[:args.limit]

        for method in methods:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] {dataset} - {method}")
            if dataset not in ["qm9test", "zinc", "qm9", "aqsol"] and method == "smiles":
                print(f"   ⚠️ 跳过: {method} 不适用于 {dataset}")
                continue
            metrics = collect_single_method_metrics(
                dataset=dataset,
                method=method,
                bpe_num_merges=args.bpe_merges,
                bpe_min_frequency=args.bpe_min_freq,
                limit=None,  # 已在 shared_graphs 上限裁剪
                mult=args.mult,
                udi=shared_udi,
                loader=shared_loader,
                graphs=shared_graphs,
            )
            all_metrics.append(metrics)
            if metrics.get('success', False):
                print(f"   ✅ 完成: {method}")
            else:
                print(f"   ❌ 失败: {method}")
            print()
    
    total_time = time.time() - start_time
    
    # 保存结果
    print("=" * 80)
    print("🎉 数据收集完成!")
    print(f"⏱️  总耗时: {total_time:.2f}s")
    
    successful_count = sum(1 for m in all_metrics if m.get('success', False))
    print(f"✅ 成功: {successful_count}/{len(all_metrics)} 个测试")
    
    if successful_count > 0:
        save_metrics_to_csv(all_metrics, output_dir)
    
    # 保存完整的原始数据
    raw_data_file = output_dir / f"efficiency_metrics_raw_{int(time.time())}.json"
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': all_metrics,
            'config': {
                'datasets': datasets,
                'methods': methods,
                'bpe_merges': args.bpe_merges,
                'bpe_min_freq': args.bpe_min_freq,
                'limit': args.limit
            },
            'summary': {
                'total_time': total_time,
                'successful_count': successful_count,
                'total_count': len(all_metrics)
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 原始数据已保存: {raw_data_file}")
    print()
    print("🎯 现在可以使用以下命令生成图表:")
    print(f"   cd {output_dir}")
    print("   python run_all_plots.py")
    print()


if __name__ == "__main__":
    main()
