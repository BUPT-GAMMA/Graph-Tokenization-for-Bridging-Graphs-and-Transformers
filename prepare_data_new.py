#!/usr/bin/env python3
"""
Data preparation script: serialization + BPE training + vocabulary building.

Performs the full data pipeline:
1. Graph serialization (supports multiple realizations per graph)
2. BPE training on all serialized sequences
3. Build and save BERT vocabulary
4. Parallel processing across multiple serialization methods
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
    """Worker function: builds environment independently in subprocess to avoid passing complex objects."""
    dataset, method, bpe_num_merges, bpe_min_frequency, multiple_samples, experiment_name, experiment_group, workers, debug = args_tuple
    method_key = f"{dataset}_{method}"
    
    try:
        # Lazy imports for clean subprocess context
        from config import ProjectConfig  # type: ignore
        from src.data.unified_data_interface import UnifiedDataInterface  # type: ignore
        from src.algorithms.compression.bpe_engine import BPEEngine  # type: ignore
        from src.models.bert.vocab_manager import build_vocab_from_sequences  # type: ignore

        # Create config
        config = ProjectConfig()
        config.dataset.name = dataset
        if experiment_name:
            config.experiment_name = experiment_name
        if experiment_group:
            config.experiment_group = experiment_group
        
        # Apply BPE config overrides
        config.serialization.bpe.num_merges = bpe_num_merges
        config.serialization.bpe.min_frequency = bpe_min_frequency
        
        # Apply multiple sampling config
        if multiple_samples is not None:
            config.serialization.multiple_sampling.num_realizations = multiple_samples
            config.serialization.multiple_sampling.enabled = multiple_samples > 1

        print(f"Processing method: {method} (dataset: {dataset})")
        
        # Create UDI instance
        udi = UnifiedDataInterface(config, dataset)
        
        # Stats collection
        stats = {
            'method': method,
            'dataset': dataset,
            'start_time': time.time(),
            'serialization_time': 0,
            'bpe_training_time': 0,
        }
        
        # 1. Ensure serialized data exists (auto-builds if missing)
        print(f"Preparing serialized data: {method}")
        serialization_start = time.time()
        serialization_path = udi.prepare_serialization(method)
        serialization_end = time.time()
        stats['serialization_time'] = serialization_end - serialization_start
        print(f"Serialization done: {serialization_path} ({stats['serialization_time']:.2f}s)")
        
        # 2. Load all sequences for BPE training
        print("Loading sequences for BPE training...")
        sequences_with_ids, properties = udi.get_sequences(method)
        
        # Extract pure sequence list (drop graph IDs)
        sequences = [seq for _, seq in sequences_with_ids]
        print(f"Got {len(sequences)} sequences for BPE training")
        
        assert sequences, f"Method {method} produced no sequences"
        
        # Serialization stats
        seq_lengths = [len(seq) for seq in sequences]
        serialization_stats = {
            'num_sequences': len(sequences),
            'avg_sequence_length': sum(seq_lengths) / len(seq_lengths),
        }
        stats['serialization_stats'] = serialization_stats
        
        print(f"Serialization stats: avg length {serialization_stats['avg_sequence_length']:.1f}")
        
        # 3. Train BPE model
        print("Training BPE model...")
        bpe_start = time.time()
        
        # Build BPE engine
        engine = BPEEngine(
            train_backend='cpp',
            encode_backend='cpp',
            encode_rank_mode='all',
        )
        
        # Train BPE
        train_stats = engine.train(
            sequences, 
            num_merges=int(bpe_num_merges),
            min_frequency=int(bpe_min_frequency)
        )
        
        bpe_end = time.time()
        stats['bpe_training_time'] = bpe_end - bpe_start
        
        # BPE compression stats
        compression_stats = {
            'num_merges_requested': int(bpe_num_merges),
            'num_merges_performed': train_stats['num_merges_performed'],
            'min_frequency': int(bpe_min_frequency),
            'final_vocab_size': engine.vocab_size,
            'training_time': stats['bpe_training_time'],
        }
        stats['compression_stats'] = compression_stats
        
        print(f"BPE training done ({stats['bpe_training_time']:.2f}s)")
        print(f"   - Merges performed: {compression_stats['num_merges_performed']}")
        print(f"   - Final vocab size: {compression_stats['final_vocab_size']}")
        
        # 4. Save BPE model (codebook only, not encoded results)
        print("Saving BPE model...")
        model_path = udi.save_bpe_codebook(method, engine.merge_rules, engine.vocab_size)
        print(f"BPE model saved: {model_path}")
        
        # Build BPE-encoded data for vocab building (batch interface is more efficient)
        bpe_encoded_sequences = engine.batch_encode(sequences)

        # DEBUG: 详细分析BPE编码效果
        unique_tokens = {int(t) for seq in bpe_encoded_sequences for t in seq}
        if debug:
            print(f"[DEBUG] unique tokens after encode = {len(unique_tokens)} (first 20: {sorted(list(unique_tokens))[:20]})")
        
        # Check BPE encoder state
        if debug:
            print("[DEBUG] BPE engine info:")
            print(f"  - encode_backend: {engine.encode_backend}")
            print(f"  - encode_rank_mode: {engine.encode_rank_mode}")
            print(f"  - vocab_size: {engine.vocab_size}")
            print(f"  - merge_rules count: {len(engine.merge_rules) if engine.merge_rules else 0}")
        
        # Test single sequence encoding in detail
        if debug and sequences:
            test_seq = sequences[0]
            print("[DEBUG] Test sequence encoding:")
            print(f"  - Original (len={len(test_seq)}): {test_seq}")
            encoded_test = engine.encode(test_seq)
            print(f"  - Encoded (len={len(encoded_test)}): {encoded_test}")
            print(f"  - Compression ratio: {len(encoded_test)/len(test_seq):.3f}")
        
        # Optional: token range distribution
        if debug:
            token_ranges = {
                "0-10": sum(1 for t in unique_tokens if 0 <= t <= 10),
                "11-100": sum(1 for t in unique_tokens if 11 <= t <= 100),
                "101-1000": sum(1 for t in unique_tokens if 101 <= t <= 1000),
                "1001+": sum(1 for t in unique_tokens if t > 1000)
            }
            print(f"[DEBUG] Token range distribution: {token_ranges}")
        
        # 5. Build and save vocabulary
        print("Building vocabulary...")
        vocab_start = time.time()
        
        # Ensure vocab covers all BPE tokens:
        # 1) base tokens = all original token IDs from training sequences
        # 2) new tokens = new_id set from merge_rules
        # Inject their union as a single sequence to avoid assuming contiguous 0..vocab_size-1
        base_token_ids = {int(t) for seq in sequences for t in seq}
        new_token_ids = {int(nid) for (_, _, nid) in engine.merge_rules}
        all_codebook_token_ids = list(base_token_ids | new_token_ids)
        bpe_encoded_sequences.append(all_codebook_token_ids)

        vocab_manager = build_vocab_from_sequences(
            bpe_encoded_sequences,
            config,
            min_freq=1,
            max_vocab_size=None
        )
        
        # Register vocab with UDI
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
        
        print(f"Vocab built ({vocab_stats['vocab_building_time']:.2f}s)")
        print(f"   - Vocab size: {vocab_stats['vocab_size']}")
        print(f"   - Vocab path: {vocab_path}")
        
        # 6. Save stats
        stats['end_time'] = time.time()
        stats['total_time'] = stats['end_time'] - stats['start_time']
        stats['model_path'] = str(model_path)
        stats['serialization_path'] = str(serialization_path)
        
        # Save stats to JSON
        stats_dir = model_path.parent / "stats"
        stats_dir.mkdir(exist_ok=True)
        stats_file = stats_dir / f"{method}_processing_stats.json"
        
        # Remove non-JSON-serializable fields
        json_stats = {k: v for k, v in stats.items() if k not in ['start_time', 'end_time']}
        json_stats['processing_time_formatted'] = f"{stats['total_time']:.2f}s"
        json_stats['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats['start_time']))
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(json_stats, f, indent=2, ensure_ascii=False)
        
        print(f"Stats saved: {stats_file}")
        
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
        
        print(f"Done {method}: {serialization_stats['num_sequences']} seqs, BPE vocab {compression_stats['final_vocab_size']}, BERT vocab {vocab_stats['vocab_size']}")
        return method_key, result
        
    except Exception:
        import traceback
        traceback.print_exc()
        return method_key, {'method': method, 'dataset': dataset, 'error': traceback.format_exc()}


def init_worker() -> None:
    """Subprocess init: ignore SIGINT so the main process handles Ctrl+C."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)





def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Data preparation: serialization + BPE training + vocabulary building")
    parser.add_argument("--datasets", type=str, default="qm9test", help="Comma-separated dataset list (default: qm9test)")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated method list; uses all available if omitted")
    parser.add_argument("--workers", type=int, default=64, help="Method-level parallelism (subprocess count)")
    parser.add_argument("--child", action="store_true", help="Child process mode: output results only, no summary")
    parser.add_argument("--bpe_merges", type=int, default=2000, help="BPE merge count")
    parser.add_argument("--bpe_min_freq", type=int, default=2, help="BPE minimum frequency threshold")
    parser.add_argument("--multiple_samples", type=int, default=None, help="Multiple sampling count per graph")
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--experiment_group", type=str, default=None, help="Experiment group")
    parser.add_argument("--out", default=None, help="Output directory (default: prepare_results)")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug info")
    
    args_ns = parser.parse_args()

    # Dataset and method lists
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

    print("Preparation config:")
    print(f"   Datasets: {datasets}")
    print(f"   Methods ({len(methods)}): {methods}")
    print(f"   BPE config: num_merges={args_ns.bpe_merges}, min_frequency={args_ns.bpe_min_freq}")
    print(f"   Workers: {args_ns.workers}")
    print(f"   Output dir: {results_dir}")

    start_time = time.time()

    results: Dict[str, Any] = {}
    # Child mode: single dataset x method, output JSON only, no summary
    if args_ns.child:
        try:
            assert datasets and len(datasets) == 1, "--child mode requires exactly one dataset"
            assert methods and len(methods) == 1, "--child mode requires exactly one method"
            dataset = datasets[0]
            method = methods[0]
            method_key, data = _prepare_one_mp((dataset, method, args_ns.bpe_merges, args_ns.bpe_min_freq, args_ns.multiple_samples, args_ns.experiment_name, args_ns.experiment_group, args_ns.workers, args_ns.debug))
            results[method_key] = data
            # Save and return directly, skip summary
            cfg_dump = {
                'datasets': [dataset],
                'methods': [method],
                'bpe_config': {'num_merges': args_ns.bpe_merges, 'min_frequency': args_ns.bpe_min_freq},
                'workers': 1,
                'multiple_samples': args_ns.multiple_samples,
            }
            results_file = results_dir / f"prepare_results_{dataset}.json"
            # Ensure directory exists (subprocess may not see parent-created dirs)
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with results_file.open('w') as f:
                json.dump({'results': results, 'config': cfg_dump, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
            # Child process ends here
            return
        except Exception:
            import traceback as _tb
            print(_tb.format_exc())
            # Return non-zero exit code for parent to detect
            sys.exit(1)

    import subprocess
    num_workers = max(1, int(args_ns.workers))
    # Build task list: dataset x method cartesian product
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
    
    # Concurrent launch with real-time output aggregation
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
            print(f"[{task_key}] Output read error:\n{traceback.format_exc()}")
    finished_order: List[str] = []
    def _start_next():
        if not pending:
            return
        task_key, cmd, cdir = pending.pop(0)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=_pump_stdout, args=(proc, task_key), daemon=True)
        t.start()
        active[task_key] = (proc, t, cdir)
    # Initial launch up to concurrency limit
    for _ in range(min(num_workers, len(pending))):
        _start_next()
    # Poll and backfill
    while active:
        to_remove = []
        for task_key, (proc, t, cdir) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                t.join(timeout=1)
                # Read result
                if rc != 0:
                    results[task_key] = {"task": task_key, "error": f"child failed ({rc})"}
                else:
                    # Extract dataset name (strip trailing method name)
                    dataset_name = '_'.join(task_key.split('_')[:-1])
                    child_json = cdir / f"prepare_results_{dataset_name}.json"
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
                        print(f"[{task_key}] Result read failed:\n{traceback.format_exc()}")
                        results[task_key] = {"task": task_key, "error": str(e)}
                finished_order.append(task_key)
                to_remove.append(task_key)
        for task_key in to_remove:
            active.pop(task_key, None)
            _start_next()
        # Short sleep to avoid busy polling
        time.sleep(0.05)

    total_time = time.time() - start_time

    # Save full results
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

    print(f"\nData preparation complete!")
    print(f"Total time: {total_time:.2f}s")

    successful_tasks = [k for k, v in results.items() if 'error' not in v]
    failed_tasks = [k for k, v in results.items() if 'error' in v]
    print(f"Success: {len(successful_tasks)}/{len(results)} tasks")
    if failed_tasks:
        print(f"Failed: {failed_tasks}")

    if successful_tasks:
        print("\nResults summary:")
        print(f"{'Task':<20} {'Status':<8} {'Seqs':<8} {'AvgLen':<10} {'BPE_Vocab':<10} {'BERT(-sp)':<12} {'BERT(+sp)':<12} {'Merges':<10} {'Time':<8}")
        print("-" * 140)
        
        for task_key in successful_tasks:
            result = results[task_key]
            if 'num_sequences' in result:
                avg_len = f"{result['avg_sequence_length']:.1f}"
                task_time = f"{result['total_time']:.1f}s"
                no_spec = int(result.get('bert_vocab_size_no_specials', int(result['bert_vocab_size']) - int(result.get('special_tokens', 8))))
                print(f"{task_key:<20} {'OK':<8} {result['num_sequences']:<8} "
                      f"{avg_len:<10} {result['bpe_vocab_size']:<10} {no_spec:<12} {result['bert_vocab_size']:<12} {result['num_merges_performed']:<10} {task_time:<8}")

        for task_key in failed_tasks:
            result = results[task_key] 
            print(f"{task_key:<20} {'FAIL':<8} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<8}")
            print(f"   Error: {result.get('error', 'unknown')}")

    print(f"\nAll results saved to: {results_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
