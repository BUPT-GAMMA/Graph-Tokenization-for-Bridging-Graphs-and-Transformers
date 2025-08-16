#!/usr/bin/env python3
"""
序列化方法+BPE压缩效果对比测试（简化版本）
=================================================

- 直接在单脚本内完成基准评测：序列化速度、序列统计、BPE训练与压缩速度、压缩效果等
- 使用统一数据接口与序列化器工厂；不做隐式回退，不读写缓存（仅评测，不持久化）
- 并行按方法运行（可选），使用多进程（spawn）；并发度由 --workers 控制
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import math

import numpy as np
import matplotlib.pyplot as plt

from config import ProjectConfig
from src.utils.visualization_helper import setup_plot_style


def generate_report(results: Dict[str, Any], results_dir: Path, bpe_config: dict, dataset: str):
    """生成测试报告（复刻旧版结构，后续可删减）"""
    report_file = results_dir / f"comparison_report_{dataset}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 序列化方法+BPE压缩效果对比报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**数据集**: {dataset}\n")
        f.write(f"**BPE配置**: {bpe_config}\n\n")

        successful_methods = {k: v for k, v in results.items() if 'error' not in v}
        #按照方法名排序
        successful_methods = dict(sorted(successful_methods.items(), key=lambda x: x[0]))

        if successful_methods:
            f.write("## 序列化效果对比\n\n")
            f.write("| 方法 | 分子数量 | 原始总长度 | 原始平均长度 | 序列化时间(s) | 序列化速度(分子/s) |\n")
            f.write("|------|----------|------------|------------|--------------|------------------|\n")
            for method_key, result in successful_methods.items():
                assert 'original_stats' in result, f"结果数据缺少 'original_stats' 字段: {method_key}"
                orig = result['original_stats']
                f.write(
                    f"| {result['method']} | "
                    f"{result['total_molecules']:,} | "
                    f"{orig['total_tokens']:,} | "
                    f"{orig['avg_length']:.1f} | "
                    f"{result['serialization_time']:.2f} | "
                    f"{result['serialization_speed']:.1f} |\n"
                )

            f.write("\n## BPE压缩效果对比\n\n")
            f.write("| 方法 | 压缩后总长度 | 压缩比 | 节省tokens | 压缩率(%) | BPE码本大小 | BPE训练时间(s) |\n")
            f.write("|------|------------|--------|-----------|-----------|------------|---------------|\n")
            for method_key, result in successful_methods.items():
                assert 'compression_stats' in result, f"结果数据缺少 'compression_stats' 字段: {method_key}"
                assert 'bpe_train_stats' in result, f"结果数据缺少 'bpe_train_stats' 字段: {method_key}"
                comp = result['compression_stats']
                train_stats = result['bpe_train_stats']
                f.write(
                    f"| {result['method']} | "
                    f"{comp['compressed_total_tokens']:,} | "
                    f"{comp['compression_ratio']:.3f} | "
                    f"{comp['tokens_saved']:,} | "
                    f"{comp['compression_percentage']:.1f}% | "
                    f"{train_stats['final_vocab_size']:,} | "
                    f"{result['bpe_train_time']:.2f} |\n"
                )

            f.write("\n## 质量评估\n\n")
            f.write("| 方法 | 解码准确率 | 测试样本数 | 压缩时间(s) |\n")
            f.write("|------|----------|----------|------------|\n")
            for method_key, result in successful_methods.items():
                f.write(
                    f"| {result['method']} | "
                    f"{result['decode_accuracy']:.1%} | "
                    f"{result['sample_size']} | "
                    f"{result['compress_time']:.2f} |\n"
                )

            f.write("\n## Pair统计信息\n\n")
            f.write("| 方法 | 总Pair数量 | 唯一Pair数量 | 平均每序列Pair数 | BPE合并次数 |\n")
            f.write("|------|-----------|------------|----------------|----------|\n")
            for method_key, result in successful_methods.items():
                assert 'pair_stats' in result, f"结果数据缺少 'pair_stats' 字段: {method_key}"
                assert 'bpe_train_stats' in result, f"结果数据缺少 'bpe_train_stats' 字段: {method_key}"
                pair_stats = result['pair_stats']
                train_stats = result['bpe_train_stats']
                f.write(
                    f"| {result['method']} | "
                    f"{pair_stats['total_pairs']:,} | "
                    f"{pair_stats['unique_pairs']:,} | "
                    f"{pair_stats['avg_pairs_per_sequence']:.1f} | "
                    f"{train_stats['num_merges_performed']} |\n"
                )

        failed_methods = {k: v for k, v in results.items() if 'error' in v}
        if failed_methods:
            f.write("\n## 失败的方法\n\n")
            for method_key, result in failed_methods.items():
                f.write(f"- **{result['method']}**: {result['error']}\n")

    print(f"📄 报告已生成: {report_file}")


def generate_visualization(results: Dict[str, Any], results_dir: Path, dataset: str):
    """生成可视化图表（复刻旧版，后续可删减）"""
    successful_methods = {k: v for k, v in results.items() if 'error' not in v}
    if not successful_methods:
        print("⚠️ 没有成功的方法用于可视化")
        return

    try:
        setup_plot_style()

        methods = list(successful_methods.keys())
        methods.sort()

        original_total_tokens = []
        compressed_total_tokens = []
        compression_ratios = []
        vocab_sizes = []
        serialization_speeds = []
        # 每序列均值±3σ（压缩前/后）
        perseq_orig_mean = []
        perseq_orig_std = []
        perseq_comp_mean = []
        perseq_comp_std = []

        for method in methods:
            result = successful_methods[method]
            assert 'original_stats' in result, f"结果数据缺少 'original_stats' 字段: {method}"
            assert 'compression_stats' in result, f"结果数据缺少 'compression_stats' 字段: {method}"
            assert 'bpe_train_stats' in result, f"结果数据缺少 'bpe_train_stats' 字段: {method}"
            
            orig_stats = result['original_stats']
            comp_stats = result['compression_stats']
            train_stats = result['bpe_train_stats']

            original_total_tokens.append(orig_stats['total_tokens'] / 1000)
            compressed_total_tokens.append(comp_stats['compressed_total_tokens'] / 1000)
            compression_ratios.append(comp_stats['compression_ratio'])
            vocab_sizes.append(train_stats['final_vocab_size'])
            serialization_speeds.append(result['serialization_speed'])
            ps = result.get('per_sequence_stats', {})
            perseq_orig_mean.append(ps.get('orig_mean', 0.0))
            perseq_orig_std.append(ps.get('orig_std', 0.0))
            perseq_comp_mean.append(ps.get('comp_mean', 0.0))
            perseq_comp_std.append(ps.get('comp_std', 0.0))

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'序列化方法+BPE压缩效果综合对比 - {dataset}', fontsize=16, fontweight='bold')

        x = np.arange(len(methods))
        width = 0.35

        bars1 = axes[0, 0].bar(x - width/2, original_total_tokens, width, label='原始长度', color='lightcoral', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, compressed_total_tokens, width, label='压缩后长度', color='lightblue', alpha=0.8)
        axes[0, 0].set_ylabel('总长度 (K tokens)')
        axes[0, 0].set_title('压缩前后总长度对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(original_total_tokens)*0.01, f'{height:.1f}K', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(compressed_total_tokens)*0.01, f'{height:.1f}K', ha='center', va='bottom', fontsize=8)

        bars3 = axes[0, 1].bar(methods, compression_ratios, color='lightgreen', alpha=0.8)
        axes[0, 1].set_ylabel('压缩比 (越小越好)')
        axes[0, 1].set_title('BPE压缩比对比')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        for bar, ratio in zip(bars3, compression_ratios):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(compression_ratios)*0.01, f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')

        # 新图：每序列长度（均值±1/2/3σ），压缩前后成对柱状
        width = 0.35
        x = np.arange(len(methods))
        axes[0, 2].bar(x - width/2, perseq_orig_mean, width, color='lightslategray', alpha=0.9, label='原始均值')
        axes[0, 2].bar(x + width/2, perseq_comp_mean, width, color='mediumseagreen', alpha=0.9, label='压缩后均值')
        # 绘制 ±1/2/3σ 的误差线（用竖直线+上下横线模拟“工”形范围）
        # 颜色深浅：1σ(浅灰) 2σ(中灰) 3σ(黑)
        sigma_colors = {1: '#bbbbbb', 2: '#777777', 3: '#000000'}
        for i in range(len(methods)):
            mu, sd = perseq_orig_mean[i], perseq_orig_std[i]
            for k in (1, 2, 3):
                lo = max(mu - k * sd, 0.0)
                hi = mu + k * sd
                axes[0, 2].vlines(x[i] - width/2, lo, hi, colors=sigma_colors[k], linewidth=2 - (0 if k == 3 else 0.5))
                axes[0, 2].hlines([lo, hi], x[i] - width/2 - width*0.15, x[i] - width/2 + width*0.15, colors=sigma_colors[k], linewidth=2 - (0 if k == 3 else 0.5))
            mu2, sd2 = perseq_comp_mean[i], perseq_comp_std[i]
            for k in (1, 2, 3):
                lo2 = max(mu2 - k * sd2, 0.0)
                hi2 = mu2 + k * sd2
                axes[0, 2].vlines(x[i] + width/2, lo2, hi2, colors=sigma_colors[k], linewidth=2 - (0 if k == 3 else 0.5))
                axes[0, 2].hlines([lo2, hi2], x[i] + width/2 - width*0.15, x[i] + width/2 + width*0.15, colors=sigma_colors[k], linewidth=2 - (0 if k == 3 else 0.5))
        axes[0, 2].set_ylabel('每序列长度 (均值±1/2/3σ)')
        axes[0, 2].set_title('每序列长度对比（压缩前后）')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(methods, rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        # 为 1/2/3σ 添加图例说明
        try:
            from matplotlib.lines import Line2D as _Line2D
            legend_lines = [
                _Line2D([0], [0], color=sigma_colors[1], lw=2, label='±1σ'),
                _Line2D([0], [0], color=sigma_colors[2], lw=1.5, label='±2σ'),
                _Line2D([0], [0], color=sigma_colors[3], lw=2, label='±3σ'),
            ]
            bars_legend = axes[0, 2].legend(loc='upper left')
            extra_legend = axes[0, 2].legend(handles=legend_lines, loc='upper right')
            axes[0, 2].add_artist(bars_legend)
            axes[0, 2].add_artist(extra_legend)
        except Exception:
            axes[0, 2].legend()

        bars5 = axes[1, 0].bar(methods, vocab_sizes, color='lightcoral', alpha=0.8)
        axes[1, 0].set_ylabel('BPE码本大小')
        axes[1, 0].set_title('BPE词汇表大小对比')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        for bar, vocab_size in zip(bars5, vocab_sizes):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(vocab_sizes)*0.01, f'{vocab_size:,}', ha='center', va='bottom', fontweight='bold')

        bars6 = axes[1, 1].bar(methods, serialization_speeds, color='lightblue', alpha=0.8)
        axes[1, 1].set_ylabel('序列化速度 (分子/秒)')
        axes[1, 1].set_title('序列化处理速度对比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        for bar, speed in zip(bars6, serialization_speeds):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(serialization_speeds)*0.01, f'{speed:.1f}', ha='center', va='bottom', fontweight='bold')

        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, 'BPE 解码准确率=100% (已强制校验)', ha='center', va='center', fontsize=12)

        plt.tight_layout()
        chart_file = results_dir / f"comprehensive_comparison_chart_{dataset}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()

        generate_detailed_charts(methods, successful_methods, results_dir, dataset)
        print(f"📊 综合图表已生成: {chart_file}")

    except Exception as e:
        print(f"❌ 可视化生成失败: {e}")
        import traceback
        traceback.print_exc()


def generate_detailed_charts(methods: list, successful_methods: dict, results_dir: Path, dataset: str):
    try:
        total_pairs = []
        unique_pairs = []
        avg_pairs_per_seq = []
        merge_counts = []

        for method in methods:
            result = successful_methods[method]
            assert 'pair_stats' in result, f"结果数据缺少 'pair_stats' 字段: {method}"
            assert 'bpe_train_stats' in result, f"结果数据缺少 'bpe_train_stats' 字段: {method}"
            pair_stats = result['pair_stats']
            train_stats = result['bpe_train_stats']
            total_pairs.append(pair_stats['total_pairs'] / 1000000)
            unique_pairs.append(pair_stats['unique_pairs'])
            avg_pairs_per_seq.append(pair_stats['avg_pairs_per_sequence'])
            merge_counts.append(train_stats['num_merges_performed'])

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'序列化方法Pair统计详细分析 - {dataset}', fontsize=16, fontweight='bold')

        bars1 = axes[0, 0].bar(methods, total_pairs, color='purple', alpha=0.8)
        axes[0, 0].set_ylabel('总Pair数量 (百万)')
        axes[0, 0].set_title('序列化方法产生的总Pair数量对比')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        for bar, count in zip(bars1, total_pairs):
            if count > 0:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(total_pairs)*0.01, f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')

        bars2 = axes[0, 1].bar(methods, unique_pairs, color='orange', alpha=0.8)
        axes[0, 1].set_ylabel('唯一Pair数量')
        axes[0, 1].set_title('序列化方法产生的唯一Pair数量对比')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        for bar, count in zip(bars2, unique_pairs):
            if count > 0:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(unique_pairs)*0.01, f'{count:,}', ha='center', va='bottom', fontweight='bold')

        bars3 = axes[1, 0].bar(methods, avg_pairs_per_seq, color='lightblue', alpha=0.8)
        axes[1, 0].set_ylabel('平均每序列Pair数量')
        axes[1, 0].set_title('序列化方法平均每序列Pair数量对比')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        for bar, count in zip(bars3, avg_pairs_per_seq):
            if count > 0:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(avg_pairs_per_seq)*0.01, f'{count:.1f}', ha='center', va='bottom', fontweight='bold')

        bars4 = axes[1, 1].bar(methods, merge_counts, color='lightgreen', alpha=0.8)
        axes[1, 1].set_ylabel('BPE合并次数')
        axes[1, 1].set_title('BPE实际执行合并次数对比')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        for bar, count in zip(bars4, merge_counts):
            if count > 0:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + max(merge_counts)*0.01, f'{count}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        detail_chart_file = results_dir / f"detailed_analysis_chart_{dataset}.png"
        plt.savefig(detail_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 详细分析图表已生成: {detail_chart_file}")
    except Exception as e:
        print(f"⚠️ 详细图表生成失败: {e}")


@dataclass
class BenchmarkArgs:
    dataset: str
    methods: List[str]
    version: str
    workers: int
    bpe_num_merges: int
    bpe_min_frequency: int
    limit: int | None
    decode_sample: int
    results_dir: Path


def _compute_pair_stats(sequences: List[List[int]]) -> Dict[str, Any]:
    total_pairs = 0
    unique_pairs: set[Tuple[int, int]] = set()
    for seq in sequences:
        if len(seq) >= 2:
            total_pairs += (len(seq) - 1)
            unique_pairs.update(zip(seq, seq[1:]))
    avg_pairs = (total_pairs / len(sequences)) if sequences else 0.0
    return {
        'total_pairs': int(total_pairs),
        'unique_pairs': int(len(unique_pairs)),
        'avg_pairs_per_sequence': float(avg_pairs),
    }


def _benchmark_one_mp(args_tuple: Tuple[str, str, int, int, int | None, int, int]) -> Tuple[str, Dict[str, Any]]:
    """子进程基准函数：避免在父进程传递复杂对象，子进程内独立构建环境。"""
    method, dataset, bpe_num_merges, bpe_min_frequency, limit, decode_sample, per_method_workers = args_tuple
    try:
        # 延迟导入，确保子进程上下文干净
        from config import ProjectConfig  # type: ignore
        from src.data.unified_data_interface import UnifiedDataInterface  # type: ignore
        from src.algorithms.serializer.serializer_factory import SerializerFactory  # type: ignore
        from src.algorithms.compression.main_bpe import StandardBPECompressor  # type: ignore

        cfg = ProjectConfig()
        cfg.dataset.name = dataset
        udi = UnifiedDataInterface(config=cfg, dataset=dataset)
        loader = udi.get_dataset_loader()
        graphs, _ = loader.get_all_data_with_indices()
        if limit is not None:
            graphs = graphs[: int(limit)]

        # 序列化评测（不持久化）
        serializer = SerializerFactory.create_serializer(method)
        # 为内部并发设置每方法 CPU 配额
        serializer.stats_num_workers = max(1, int(per_method_workers))
        serializer.initialize_with_dataset(loader, graphs)

        t0 = time.perf_counter()
        batch_results = serializer.batch_serialize(graphs, desc=f"serialize-{method}")
        t1 = time.perf_counter()

        sequences: List[List[int]] = []
        for res in batch_results:
            if res and getattr(res, 'token_sequences', None):
                sequences.append(res.token_sequences[0])
            else:
                sequences.append([])

        sequences_nonempty = [s for s in sequences if len(s) > 0]
        assert sequences_nonempty, "序列化未产生有效序列"

        total_tokens = int(sum(len(s) for s in sequences_nonempty))
        avg_length = float(total_tokens / len(sequences_nonempty))
        serialization_time = max(t1 - t0, 1e-9)
        serialization_speed = len(graphs) / serialization_time if serialization_time > 0 else 0.0

        compressor = StandardBPECompressor(num_merges=int(bpe_num_merges), min_frequency=int(bpe_min_frequency), debug=False)
        t2 = time.perf_counter()
        bpe_train_stats = compressor.train(sequences_nonempty)
        t3 = time.perf_counter()
        bpe_train_time = max(t3 - t2, 1e-9)

        t4 = time.perf_counter()
        encoded = [compressor.encode(seq) for seq in sequences_nonempty]
        t5 = time.perf_counter()
        compress_time = max(t5 - t4, 1e-9)

        compressed_total_tokens = int(sum(len(s) for s in encoded))
        compression_ratio = (compressed_total_tokens / total_tokens) if total_tokens > 0 else 1.0
        compression_percentage = (1.0 - compression_ratio) * 100.0
        tokens_saved = total_tokens - compressed_total_tokens

        sample_size = min(len(sequences_nonempty), int(decode_sample))
        correct = 0
        for i in range(sample_size):
            dec = compressor.decode(encoded[i])
            if dec == sequences_nonempty[i]:
                correct += 1
        decode_accuracy = (correct / sample_size) if sample_size > 0 else 0.0
        # 严格约束：BPE 解码必须完全准确
        assert correct == sample_size, f"BPE 解码失败: {correct}/{sample_size}"

        pair_stats = _compute_pair_stats(sequences_nonempty)

        # 每序列长度统计（均值/标准差），用于可视化的误差线（±3σ）
        def _mean_std(arr_lens: List[int]) -> Tuple[float, float]:
            if not arr_lens:
                return 0.0, 0.0
            n = float(len(arr_lens))
            mean = float(sum(arr_lens) / n)
            if n <= 1:
                return mean, 0.0
            var = float(sum((x - mean) ** 2 for x in arr_lens) / (n - 1))
            return mean, math.sqrt(var)

        orig_len_list = [len(s) for s in sequences_nonempty]
        comp_len_list = [len(s) for s in encoded]
        orig_mean, orig_std = _mean_std(orig_len_list)
        comp_mean, comp_std = _mean_std(comp_len_list)

        result = {
            'method': method,
            'total_molecules': len(graphs),
            'original_stats': {'total_tokens': total_tokens, 'avg_length': avg_length},
            'serialization_time': serialization_time,
            'serialization_speed': serialization_speed,
            'bpe_train_time': bpe_train_time,
            'bpe_train_stats': bpe_train_stats,
            'compression_stats': {
                'compressed_total_tokens': compressed_total_tokens,
                'compression_ratio': compression_ratio,
                'tokens_saved': tokens_saved,
                'compression_percentage': compression_percentage,
            },
            'compress_time': compress_time,
            'decode_accuracy': decode_accuracy,
            'sample_size': sample_size,
            'pair_stats': pair_stats,
            'per_sequence_stats': {
                'orig_mean': orig_mean,
                'orig_std': orig_std,
                'comp_mean': comp_mean,
                'comp_std': comp_std,
            },
        }
        print(f"✅ {method}: 序列化 {serialization_speed:.1f} 分子/s, 压缩率 {compression_ratio:.3f}")
        return method, result
    except Exception:
        import traceback
        traceback.print_exc()
        return method, {'method': method, 'error': traceback.format_exc()}


def init_worker() -> None:
    """子进程初始化：忽略 Ctrl+C 由主进程统一处理。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_for_dataset(args_ns, dataset: str) -> None:
    config = ProjectConfig()
    config.dataset.name = dataset

    # 方法列表
    if args_ns.methods:
        methods: List[str] = [m.strip() for m in args_ns.methods.split(",") if m.strip()]
    else:
        from src.algorithms.serializer.serializer_factory import SerializerFactory
        methods = SerializerFactory.get_available_serializers()

    results_dir = Path(args_ns.out or f"comparison_results/{dataset}")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("📋 测试配置:")
    print(f"   数据集: {dataset}")
    print(f"   方法数量: {len(methods)} -> {methods}")
    print(f"   BPE配置: num_merges={args_ns.bpe_num_merges}, min_frequency={args_ns.bpe_min_frequency}")
    print(f"   并行工作数: {args_ns.workers}")
    print(f"   样本上限: {args_ns.limit}")
    print(f"   解码抽样: {args_ns.decode_sample}")
    print(f"   结果目录: {results_dir}")

    bench_args = BenchmarkArgs(
        dataset=dataset,
        methods=methods,
        version=args_ns.version,
        workers=int(args_ns.workers),
        bpe_num_merges=int(args_ns.bpe_num_merges),
        bpe_min_frequency=int(args_ns.bpe_min_frequency),
        limit=args_ns.limit,
        decode_sample=int(args_ns.decode_sample),
        results_dir=results_dir,
    )

    start_time = time.time()

    results: Dict[str, Any] = {}
    # 子进程简化模式：单方法、只产出 JSON，不生成汇总图表
    if args_ns.child:
        try:
            assert methods and len(methods) == 1, "--child 模式必须指定且仅指定一个方法"
            method = methods[0]
            import os as _os
            per_method_workers = int(args_ns.per_method_workers) if args_ns.per_method_workers else max(1, (_os.cpu_count() or 1))
            method_key, data = _benchmark_one_mp((method, bench_args.dataset, bench_args.bpe_num_merges, bench_args.bpe_min_frequency, bench_args.limit, bench_args.decode_sample, per_method_workers))
            results[method_key] = data
            # 保存并直接返回，不做后续汇总输出
            cfg_dump = {
                'methods': [method],
                'dataset': bench_args.dataset,
                'bpe_config': {'num_merges': bench_args.bpe_num_merges, 'min_frequency': bench_args.bpe_min_frequency},
                'workers': 1,
                'limit': bench_args.limit,
                'decode_sample': bench_args.decode_sample,
            }
            results_file = results_dir / f"full_results_{bench_args.dataset}.json"
            with results_file.open('w') as f:
                json.dump({'results': results, 'config': cfg_dump, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
            # 子进程到此结束
            return
        except Exception:
            import traceback as _tb
            print(_tb.format_exc())
            # 异常依旧返回非零码供父进程感知
            sys.exit(1)

    import subprocess
    import os
    import threading
    num_methods_workers = max(1, int(args_ns.workers))
    per_method_workers = int(args_ns.per_method_workers) if args_ns.per_method_workers else max(1, (os.cpu_count() or 1) // num_methods_workers)
    # 预构建任务
    tasks: List[Tuple[str, List[str], Path]] = []
    for m in methods:
        child_out = results_dir / f"method_{m}"
        child_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset", dataset,
            "--methods", m,
            "--workers", "1",
            "--bpe_num_merges", str(args_ns.bpe_num_merges),
            "--bpe_min_frequency", str(args_ns.bpe_min_frequency),
            "--decode_sample", str(args_ns.decode_sample),
            "--out", str(child_out),
            "--per_method_workers", str(per_method_workers),
            "--child",
        ]
        if args_ns.limit is not None:
            cmd += ["--limit", str(args_ns.limit)]
        tasks.append((m, cmd, child_out))
    # 并发启动与实时聚合输出
    active: Dict[str, Tuple[subprocess.Popen, threading.Thread, Path]] = {}
    pending = list(tasks)
    def _pump_stdout(proc: subprocess.Popen, method_key: str):
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                print(f"[{dataset}:{method_key}] {line}")
        except Exception:
            import traceback
            print(f"[{dataset}:{method_key}] 输出读取异常:\n{traceback.format_exc()}")
    finished_order: List[str] = []
    def _start_next():
        if not pending:
            return
        m, cmd, cdir = pending.pop(0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=_pump_stdout, args=(proc, m), daemon=True)
        t.start()
        active[m] = (proc, t, cdir)
    # 初始启动至并发上限
    for _ in range(min(num_methods_workers, len(pending))):
        _start_next()
    # 轮询等待并补位
    while active:
        to_remove = []
        for m, (proc, t, cdir) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                t.join(timeout=1)
                # 读取结果
                if rc != 0:
                    results[m] = {"method": m, "error": f"child failed ({rc})"}
                else:
                    child_json = cdir / f"full_results_{dataset}.json"
                    try:
                        with child_json.open('r') as f:
                            child = json.load(f)
                        if isinstance(child, dict) and 'results' in child:
                            if m in child['results']:
                                results[m] = child['results'][m]
                            else:
                                results[m] = {"method": m, "error": "missing"}
                        else:
                            results[m] = {"method": m, "error": "malformed child json"}
                    except Exception as ex:
                        import traceback
                        print(f"[{dataset}:{m}] 结果读取失败:\n{traceback.format_exc()}")
                        results[m] = {"method": m, "error": str(ex)}
                finished_order.append(m)
                to_remove.append(m)
        for m in to_remove:
            active.pop(m, None)
            _start_next()
        # 小睡以避免忙轮询
        time.sleep(0.05)


    total_time = time.time() - start_time

    # 保存完整结果
    cfg_dump = {
        'methods': methods,
        'dataset': bench_args.dataset,
        'bpe_config': {'num_merges': bench_args.bpe_num_merges, 'min_frequency': bench_args.bpe_min_frequency},
        'workers': bench_args.workers,
        'limit': bench_args.limit,
        'decode_sample': bench_args.decode_sample,
    }
    results_file = results_dir / f"full_results_{bench_args.dataset}.json"
    with results_file.open('w') as f:
        json.dump({'results': results, 'config': cfg_dump, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'total_time': total_time}, f, indent=2)

    print("\n🎉 基准测试完成!")
    print(f"⏱️  总耗时: {total_time:.2f}s")

    successful_methods = [k for k, v in results.items() if 'error' not in v]
    failed_methods = [k for k, v in results.items() if 'error' in v]
    print(f"✅ 成功: {len(successful_methods)}/{len(methods)} 个方法")
    if failed_methods:
        print(f"❌ 失败: {failed_methods}")

    if successful_methods:
        best_method = None
        best_ratio = 1.0
        for k in successful_methods:
            assert 'compression_stats' in results[k], f"结果缺少 compression_stats: {k}"
            assert 'compression_ratio' in results[k]['compression_stats'], f"compression_stats 缺少 compression_ratio: {k}"
            cr = results[k]['compression_stats']['compression_ratio']
            if cr < best_ratio:
                best_ratio = cr
                best_method = k
        if best_method:
            print(f"🏆 最佳压缩方法: {best_method} (压缩比: {best_ratio:.3f})")

    print("\n📊 生成报告和可视化...")
    generate_report(results, results_dir, cfg_dump['bpe_config'], bench_args.dataset)
    generate_visualization(results, results_dir, bench_args.dataset)

    print(f"💾 所有结果已保存到: {results_dir}/")
    print("🎉 测试完成!")


def main():
    parser = argparse.ArgumentParser(description="序列化方法+BPE压缩效果对比测试（简化版）")
    parser.add_argument("--dataset", default="qm9test", help="数据集名称，默认 qm9test")
    parser.add_argument("--methods", default=None, help="逗号分隔的方法列表；未提供则使用全部可用方法")
    parser.add_argument("--version", default="latest", help="processed/<dataset>/<version> 目录名，默认 latest")
    parser.add_argument("--workers", type=int, default=64, help="方法级并发数（用于子进程或线程并行）")
    parser.add_argument("--per_method_workers", type=int, default=None, help="单方法内部 batch 并发度；为空则按总核数/方法数均分")
    parser.add_argument("--child", action="store_true", help="子进程模式：仅输出方法结果，不生成汇总报告与图表")
    parser.add_argument("--bpe_num_merges", type=int, default=3000, help="BPE 合并次数")
    parser.add_argument("--bpe_min_frequency", type=int, default=100, help="BPE 最小频率阈值")
    parser.add_argument("--limit", type=int, default=None, help="仅用于基准测试的样本上限（不传则全量）")
    parser.add_argument("--decode_sample", type=int, default=2000, help="解码准确率抽样数量")
    parser.add_argument("--out", default=None, help="结果输出目录；默认 comparison_results/<dataset>")

    args_ns = parser.parse_args()

    datasets: List[str] = [d.strip() for d in args_ns.dataset.split(',') if d.strip()]

    for ds in datasets:
        print("\n" + "#"*96)
        print(f"▶️ 开始数据集: {ds}")
        print("#"*96)
        _run_for_dataset(args_ns, ds)


if __name__ == "__main__":
    main()


