#!/usr/bin/env python3
"""
完整数据预处理脚本 - 序列化 + BPE训练 + 词表构建
=================================================

功能：
1. 数据序列化（支持multiple parallelization，对一个图做多次序列化）
2. 对所有序列化数据进行BPE训练
3. 构建并保存BERT词表（基于原始序列）
4. 支持多方法并行处理

设计原则：
- 基于UDI的完整实现，利用现有的序列化接口
- 支持多种序列化方法的并行批量处理
- 生成预训练所需的全部数据工件
- 遵循"无隐式回退"原则，所有操作显式进行
- 采用与run_serialization_bpe_comparison_simple.py相同的并发模式
"""

from __future__ import annotations

import argparse
import time
import json
import sys
import subprocess
import threading
from typing import List, Dict, Any, Tuple
from pathlib import Path


def _prepare_one_mp(args_tuple: Tuple[str, str, int, int, int | None, str, str, int, bool]) -> Tuple[str, Dict[str, Any]]:
    """子进程基准函数：避免在父进程传递复杂对象，子进程内独立构建环境。"""
    dataset, method, bpe_num_merges, bpe_min_frequency, multiple_samples, experiment_name, experiment_group, workers, debug = args_tuple
    method_key = f"{dataset}_{method}"
    
    try:
        # 延迟导入，确保子进程上下文干净
        from config import ProjectConfig  # type: ignore
        from src.data.unified_data_interface import UnifiedDataInterface  # type: ignore
        from src.algorithms.compression.bpe_engine import BPEEngine  # type: ignore
        from src.models.bert.vocab_manager import build_vocab_from_sequences  # type: ignore

        # 创建配置
        config = ProjectConfig()
        config.dataset.name = dataset
        if experiment_name:
            config.experiment_name = experiment_name
        if experiment_group:
            config.experiment_group = experiment_group
        
        # 应用BPE配置覆盖
        config.serialization.bpe.num_merges = bpe_num_merges
        config.serialization.bpe.min_frequency = bpe_min_frequency
        
        # 应用多重采样配置
        if multiple_samples is not None:
            config.serialization.multiple_sampling.num_realizations = multiple_samples
            config.serialization.multiple_sampling.enabled = multiple_samples > 1

        print(f"🚀 开始处理方法: {method} (数据集: {dataset})")
        
        # 创建UDI实例
        udi = UnifiedDataInterface(config, dataset)
        
        # 统计信息收集
        stats = {
            'method': method,
            'dataset': dataset,
            'start_time': time.time(),
            'serialization_time': 0,
            'bpe_training_time': 0,
        }
        
        # 1. 确保序列化数据存在（如果不存在会自动构建）
        print(f"📊 准备序列化数据: {method}")
        serialization_start = time.time()
        serialization_path = udi.prepare_serialization(method)
        serialization_end = time.time()
        stats['serialization_time'] = serialization_end - serialization_start
        print(f"✅ 序列化完成: {serialization_path} (耗时: {stats['serialization_time']:.2f}s)")
        
        # 2. 获取所有序列数据用于BPE训练
        print("📂 加载序列数据用于BPE训练...")
        sequences_with_ids, properties = udi.get_sequences(method)
        
        # 提取纯序列列表（去掉图ID）
        sequences = [seq for _, seq in sequences_with_ids]
        print(f"📊 获得 {len(sequences)} 个序列用于BPE训练")
        
        assert sequences, f"方法 {method} 没有产生任何序列"
        
        # 序列化统计（保留必要统计）
        seq_lengths = [len(seq) for seq in sequences]
        serialization_stats = {
            'num_sequences': len(sequences),
            'avg_sequence_length': sum(seq_lengths) / len(seq_lengths),
        }
        stats['serialization_stats'] = serialization_stats
        
        print(f"📊 序列化统计: 平均长度 {serialization_stats['avg_sequence_length']:.1f}")
        
        # 3. 训练BPE模型
        print("🎓 开始训练BPE模型...")
        bpe_start = time.time()
        
        # 构建BPE引擎
        engine = BPEEngine(
            train_backend='cpp',
            encode_backend='cpp',
            encode_rank_mode='all',
        )
        
        # 训练BPE
        train_stats = engine.train(
            sequences, 
            num_merges=int(bpe_num_merges),
            min_frequency=int(bpe_min_frequency)
        )
        
        bpe_end = time.time()
        stats['bpe_training_time'] = bpe_end - bpe_start
        
        # BPE压缩统计
        compression_stats = {
            'num_merges_requested': int(bpe_num_merges),
            'num_merges_performed': train_stats['num_merges_performed'],
            'min_frequency': int(bpe_min_frequency),
            'final_vocab_size': engine.vocab_size,
            'training_time': stats['bpe_training_time'],
        }
        stats['compression_stats'] = compression_stats
        
        print(f"✅ BPE训练完成 (耗时: {stats['bpe_training_time']:.2f}s)")
        print(f"   - 执行的合并次数: {compression_stats['num_merges_performed']}")
        print(f"   - 最终词汇大小: {compression_stats['final_vocab_size']}")
        
        # 4. 保存BPE模型（只保存codebook，不保存编码结果）
        print("💾 保存BPE模型...")
        model_path = udi.save_bpe_codebook(method, engine.merge_rules, engine.vocab_size)
        print(f"✅ BPE模型已保存: {model_path}")
        
        # 构建bpe编码后的数据以供构建词表（批量接口更高效）
        bpe_encoded_sequences = engine.batch_encode(sequences)

        # DEBUG: 详细分析BPE编码效果
        unique_tokens = {int(t) for seq in bpe_encoded_sequences for t in seq}
        if debug:
            print(f"[DEBUG] unique tokens after encode = {len(unique_tokens)} (示例前20: {sorted(list(unique_tokens))[:20]})")
        
        # 检查BPE编码器状态
        if debug:
            print("[DEBUG] BPE engine info:")
            print(f"  - encode_backend: {engine.encode_backend}")
            print(f"  - encode_rank_mode: {engine.encode_rank_mode}")
            print(f"  - vocab_size: {engine.vocab_size}")
            print(f"  - merge_rules count: {len(engine.merge_rules) if engine.merge_rules else 0}")
        
        # 测试单个序列的详细编码过程
        if debug and sequences:
            test_seq = sequences[0]
            print("[DEBUG] Test sequence encoding:")
            print(f"  - Original (len={len(test_seq)}): {test_seq}")
            encoded_test = engine.encode(test_seq)
            print(f"  - Encoded (len={len(encoded_test)}): {encoded_test}")
            print(f"  - Compression ratio: {len(encoded_test)/len(test_seq):.3f}")
        
        # 可选：token范围分布
        if debug:
            token_ranges = {
                "0-10": sum(1 for t in unique_tokens if 0 <= t <= 10),
                "11-100": sum(1 for t in unique_tokens if 11 <= t <= 100),
                "101-1000": sum(1 for t in unique_tokens if 101 <= t <= 1000),
                "1001+": sum(1 for t in unique_tokens if t > 1000)
            }
            print(f"[DEBUG] Token range distribution: {token_ranges}")
        
        # 5. 构建和保存词表
        print("📚 构建词表...")
        vocab_start = time.time()
        
        # 为保证词表覆盖全部 BPE token：
        # 1) 基础 token = 训练序列中出现过的所有原始 token ID
        # 2) 新 token = merge_rules 中产生的 new_id 集合
        # 将两者并集作为一次性序列注入，避免依赖“0..vocab_size-1 连续编号”的错误假设
        base_token_ids = {int(t) for seq in sequences for t in seq}
        new_token_ids = {int(nid) for (_, _, nid) in engine.merge_rules}
        all_codebook_token_ids = list(base_token_ids | new_token_ids)
        bpe_encoded_sequences.append(all_codebook_token_ids)

        vocab_manager = build_vocab_from_sequences(
            bpe_encoded_sequences,
            config,
            min_freq=1,  # 基础词频要求
            max_vocab_size=None  # 不限制词表大小
        )
        
        # 注册词表到UDI
        vocab_path = udi.register_vocab(vocab_manager, method)
        vocab_end = time.time()
        
        special_tokens_count = len(getattr(vocab_manager, 'special_tokens', []))
        vocab_stats = {
            'vocab_size': vocab_manager.vocab_size,
            'vocab_size_excl_specials': int(vocab_manager.vocab_size) - int(special_tokens_count),
            'special_tokens': int(special_tokens_count),
            'vocab_building_time': vocab_end - vocab_start,
            'vocab_path': str(vocab_path)
        }
        stats['vocab_stats'] = vocab_stats
        
        print(f"✅ 词表构建完成 (耗时: {vocab_stats['vocab_building_time']:.2f}s)")
        print(f"   - 词表大小: {vocab_stats['vocab_size']}")
        print(f"   - 词表路径: {vocab_path}")
        
        # 6. 保存统计信息
        stats['end_time'] = time.time()
        stats['total_time'] = stats['end_time'] - stats['start_time']
        stats['model_path'] = str(model_path)
        stats['serialization_path'] = str(serialization_path)
        
        # 保存统计信息到JSON文件
        stats_dir = model_path.parent / "stats"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / f"{method}_processing_stats.json"
        
        # 移除不可JSON序列化的字段
        json_stats = {k: v for k, v in stats.items() if k not in ['start_time', 'end_time']}
        json_stats['processing_time_formatted'] = f"{stats['total_time']:.2f}s"
        json_stats['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats['start_time']))
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 统计信息已保存: {stats_file}")
        
        result = {
            'method': method,
            'dataset': dataset,
            'success': True,
            'num_sequences': serialization_stats['num_sequences'],
            'avg_sequence_length': serialization_stats['avg_sequence_length'],
            'bpe_vocab_size': compression_stats['final_vocab_size'],
            'bert_vocab_size': vocab_stats['vocab_size'],
            'bert_vocab_size_no_specials': vocab_stats['vocab_size_excl_specials'],
            'special_tokens': vocab_stats['special_tokens'],
            'num_merges_performed': compression_stats['num_merges_performed'],
            'serialization_time': stats['serialization_time'],
            'bpe_training_time': stats['bpe_training_time'],
            'vocab_building_time': vocab_stats['vocab_building_time'],
            'total_time': stats['total_time'],
            'model_path': str(model_path),
            'vocab_path': str(vocab_path),
            'serialization_path': str(serialization_path),
            'stats_file': str(stats_file)
        }
        
        print(f"✅ {method}: 序列化 {serialization_stats['num_sequences']} 序列, BPE词汇 {compression_stats['final_vocab_size']}, BERT词汇 {vocab_stats['vocab_size']}")
        return method_key, result
        
    except Exception:
        import traceback
        traceback.print_exc()
        return method_key, {'method': method, 'dataset': dataset, 'error': traceback.format_exc()}


def init_worker() -> None:
    """子进程初始化：忽略 Ctrl+C 由主进程统一处理。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)





def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description="完整数据预处理脚本 - 序列化 + BPE训练 + 词表构建（简化版）")
    parser.add_argument("--datasets", type=str, default="qm9test", help="逗号分隔的数据集列表，默认 qm9test")
    parser.add_argument("--methods", type=str, default=None, help="逗号分隔的方法列表；未提供则使用全部可用方法")
    parser.add_argument("--workers", type=int, default=64, help="方法级并发数（用于子进程或线程并行）")
    parser.add_argument("--child", action="store_true", help="子进程模式：仅输出方法结果，不生成汇总报告")
    parser.add_argument("--bpe_merges", type=int, default=2000, help="BPE 合并次数")
    parser.add_argument("--bpe_min_freq", type=int, default=2, help="BPE 最小频率阈值")
    parser.add_argument("--multiple_samples", type=int, default=None, help="每个图的多重采样次数")
    parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
    parser.add_argument("--experiment_group", type=str, default=None, help="实验分组")
    parser.add_argument("--out", default=None, help="结果输出目录；默认 prepare_results")
    parser.add_argument("--debug", action="store_true", help="打印详细调试信息")
    
    args_ns = parser.parse_args()

    # 数据集和方法列表
    if args_ns.datasets:
        datasets: List[str] = [d.strip() for d in args_ns.datasets.split(",") if d.strip()]
    else:
        datasets = ["qm9test"]
        
    

    if args_ns.methods:
        methods: List[str] = [m.strip() for m in args_ns.methods.split(",") if m.strip()]
    else:
        from src.algorithms.serializer import SerializerFactory
        methods = SerializerFactory.get_available_serializers()

    results_dir = Path(args_ns.out or "prepare_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("📋 预处理配置:")
    print(f"   数据集: {datasets}")
    print(f"   方法数量: {len(methods)} -> {methods}")
    print(f"   BPE配置: num_merges={args_ns.bpe_merges}, min_frequency={args_ns.bpe_min_freq}")
    print(f"   并行工作数: {args_ns.workers}")
    print(f"   结果目录: {results_dir}")

    start_time = time.time()

    results: Dict[str, Any] = {}
    # 子进程模式：单数据集×方法、只产出 JSON，不生成汇总
    if args_ns.child:
        try:
            assert datasets and len(datasets) == 1, "--child 模式必须指定且仅指定一个数据集"
            assert methods and len(methods) == 1, "--child 模式必须指定且仅指定一个方法"
            dataset = datasets[0]
            method = methods[0]
            method_key, data = _prepare_one_mp((dataset, method, args_ns.bpe_merges, args_ns.bpe_min_freq, args_ns.multiple_samples, args_ns.experiment_name, args_ns.experiment_group, args_ns.workers, args_ns.debug))
            results[method_key] = data
            # 保存并直接返回，不做后续汇总输出
            cfg_dump = {
                'datasets': [dataset],
                'methods': [method],
                'bpe_config': {'num_merges': args_ns.bpe_merges, 'min_frequency': args_ns.bpe_min_freq},
                'workers': 1,
                'multiple_samples': args_ns.multiple_samples,
            }
            results_file = results_dir / f"prepare_results_{dataset}.json"
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
    num_workers = max(1, int(args_ns.workers))
    # 预构建任务：数据集×方法的笛卡尔积
    tasks: List[Tuple[str, List[str], Path]] = []
    for dataset in datasets:
        for method in methods:
            if dataset not in ["qm9test", "zinc","qm9","aqsol"] and method == "smiles":
                continue
            task_key = f"{dataset}_{method}"
            child_out = results_dir / f"task_{task_key}"
            child_out.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--datasets", dataset,
                "--methods", method,
                "--workers", "1",
                "--bpe_merges", str(args_ns.bpe_merges),
                "--bpe_min_freq", str(args_ns.bpe_min_freq),
                "--out", str(child_out),
                "--child",
            ]
            if args_ns.debug:
                cmd.append("--debug")
            if args_ns.multiple_samples is not None:
                cmd += ["--multiple_samples", str(args_ns.multiple_samples)]
            if args_ns.experiment_name:
                cmd += ["--experiment_name", args_ns.experiment_name]
            if args_ns.experiment_group:
                cmd += ["--experiment_group", args_ns.experiment_group]
            tasks.append((task_key, cmd, child_out))
    
    # 并发启动与实时聚合输出
    active: Dict[str, Tuple[subprocess.Popen, threading.Thread, Path]] = {}
    pending = list(tasks)
    def _pump_stdout(proc: subprocess.Popen, task_key: str):
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                print(f"[{task_key}] {line}")
        except Exception:
            import traceback
            print(f"[{task_key}] 输出读取异常:\n{traceback.format_exc()}")
    finished_order: List[str] = []
    def _start_next():
        if not pending:
            return
        task_key, cmd, cdir = pending.pop(0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=_pump_stdout, args=(proc, task_key), daemon=True)
        t.start()
        active[task_key] = (proc, t, cdir)
    # 初始启动至并发上限
    for _ in range(min(num_workers, len(pending))):
        _start_next()
    # 轮询等待并补位
    while active:
        to_remove = []
        for task_key, (proc, t, cdir) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                t.join(timeout=1)
                # 读取结果
                if rc != 0:
                    results[task_key] = {"task": task_key, "error": f"child failed ({rc})"}
                else:
                    child_json = cdir / f"prepare_results_{task_key.split('_')[0]}.json"
                    try:
                        with child_json.open('r') as f:
                            child = json.load(f)
                        if isinstance(child, dict) and 'results' in child:
                            if task_key in child['results']:
                                results[task_key] = child['results'][task_key]
                            else:
                                results[task_key] = {"task": task_key, "error": "missing"}
                        else:
                            results[task_key] = {"task": task_key, "error": "malformed child json"}
                    except Exception as e:
                        import traceback
                        print(f"[{task_key}] 结果读取失败:\n{traceback.format_exc()}")
                        results[task_key] = {"task": task_key, "error": str(e)}
                finished_order.append(task_key)
                to_remove.append(task_key)
        for task_key in to_remove:
            active.pop(task_key, None)
            _start_next()
        # 小睡以避免忙轮询
        time.sleep(0.05)

    total_time = time.time() - start_time

    # 保存完整结果
    cfg_dump = {
        'datasets': datasets,
        'methods': methods,
        'bpe_config': {'num_merges': args_ns.bpe_merges, 'min_frequency': args_ns.bpe_min_freq},
        'workers': args_ns.workers,
        'multiple_samples': args_ns.multiple_samples,
    }
    results_file = results_dir / "prepare_results_full.json"
    with results_file.open('w') as f:
        json.dump({'results': results, 'config': cfg_dump, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'total_time': total_time}, f, indent=2)

    print("\n🎉 数据预处理完成!")
    print(f"⏱️  总耗时: {total_time:.2f}s")

    successful_tasks = [k for k, v in results.items() if 'error' not in v]
    failed_tasks = [k for k, v in results.items() if 'error' in v]
    print(f"✅ 成功: {len(successful_tasks)}/{len(results)} 个任务")
    if failed_tasks:
        print(f"❌ 失败: {failed_tasks}")

    if successful_tasks:
        print("\n📊 处理结果汇总:")
        print(f"{'任务':<20} {'状态':<8} {'序列数':<8} {'平均长度':<10} {'BPE词汇':<10} {'BERT(无特)':<12} {'BERT(含特)':<12} {'合并次数':<10} {'耗时':<8}")
        print("-" * 140)
        
        for task_key in successful_tasks:
            result = results[task_key]
            if 'num_sequences' in result:
                avg_len = f"{result['avg_sequence_length']:.1f}"
                task_time = f"{result['total_time']:.1f}s"
                no_spec = int(result.get('bert_vocab_size_no_specials', int(result['bert_vocab_size']) - int(result.get('special_tokens', 8))))
                print(f"{task_key:<20} {'✅成功':<8} {result['num_sequences']:<8} "
                      f"{avg_len:<10} {result['bpe_vocab_size']:<10} {no_spec:<12} {result['bert_vocab_size']:<12} {result['num_merges_performed']:<10} {task_time:<8}")

        for task_key in failed_tasks:
            result = results[task_key] 
            print(f"{task_key:<20} {'❌失败':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8}")
            print(f"   错误: {result.get('error', 'unknown')}")

    print(f"\n💾 所有结果已保存到: {results_dir}/")
    print("🎉 处理完成!")


if __name__ == "__main__":
    main()
