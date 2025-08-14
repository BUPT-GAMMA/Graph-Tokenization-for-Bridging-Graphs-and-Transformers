#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基准：batch_serialize 串行 vs 多进程（fork）

示例：
  python scripts/benchmark_batch_serialize_mp.py --dataset qm9test --method feuler --limit 2000 --warmup 20 --workers 64

输出：
  - 串行与并行（多进程）耗时、样本吞吐、token 吞吐与速度提升倍数
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# 确保项目根目录在 sys.path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# 强制单线程（确保库内不抢线程），在导入 torch/dgl 前设置
ENV_THREADS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "DGL_NUM_THREADS": "1",
}
for k, v in ENV_THREADS.items():
    os.environ[k] = v

import torch  # noqa: E402
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)

from config import ProjectConfig  # noqa: E402
from src.data.unified_data_factory import UnifiedDataFactory  # noqa: E402
from src.algorithms.serializer.serializer_factory import SerializerFactory  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark batch_serialize serial vs multiprocessing")
    ap.add_argument("--dataset", default="qm9test", choices=["qm9", "qm9test", "zinc", "aqsol", "mnist"], help="数据集")
    ap.add_argument("--method", default="feuler", choices=["feuler", "eulerian", "cpp", "fcpp", "dfs", "bfs", "topo", "smiles"], help="序列化方法")
    ap.add_argument("--limit", type=int, default=2000, help="样本上限（按顺序截取，扣除warmup）")
    ap.add_argument("--warmup", type=int, default=20, help="预热样本数")
    ap.add_argument("--workers", type=int, default=64, help="并行进程数（fork）")
    return ap.parse_args()


def take_first_n(data: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return []
    return data[: min(n, len(data))]


def count_tokens(results) -> int:
    total = 0
    for r in results:
        if r and getattr(r, 'token_sequences', None):
            total += len(r.token_sequences[0])
    return int(total)


def run_once(dataset: str, method: str, limit: int, warmup: int, workers: int):
    cfg = ProjectConfig()
    cfg.dataset.name = dataset

    loader = UnifiedDataFactory.create(cfg.dataset.name, cfg)
    all_graphs, _ = loader.get_all_data_with_indices()

    serializer = SerializerFactory.create_serializer(method)
    # feuler/fcpp 需要全局统计：直接用全量列表初始化
    try:
        serializer.initialize_with_dataset(loader, graph_data_list=all_graphs)
    except TypeError:
        serializer.initialize_with_dataset(loader)

    warm_list = take_first_n(all_graphs, warmup)
    bench_list = take_first_n(all_graphs[warmup:], limit)

    # 预热（单次 serialize，避免把初始化成本计入基准）
    for sample in warm_list:
        _ = serializer.serialize(sample)

    # 串行（batch_serialize parallel=False）
    t0 = time.perf_counter()
    res_serial = serializer.batch_serialize(bench_list, parallel=False)
    dt_serial = time.perf_counter() - t0
    tok_serial = count_tokens(res_serial)

    # 多进程（batch_serialize parallel=True）
    t1 = time.perf_counter()
    res_mp = serializer.batch_serialize(bench_list, parallel=True, max_workers=workers)
    dt_mp = time.perf_counter() - t1
    tok_mp = count_tokens(res_mp)

    return {
        "serial": {
            "elapsed": dt_serial,
            "samples": len(res_serial),
            "tokens": tok_serial,
            "samples_s": len(res_serial) / dt_serial if dt_serial > 0 else 0.0,
            "tokens_s": tok_serial / dt_serial if dt_serial > 0 else 0.0,
        },
        "mp": {
            "elapsed": dt_mp,
            "samples": len(res_mp),
            "tokens": tok_mp,
            "samples_s": len(res_mp) / dt_mp if dt_mp > 0 else 0.0,
            "tokens_s": tok_mp / dt_mp if dt_mp > 0 else 0.0,
            "workers": workers,
        }
    }


def main():
    args = parse_args()
    r = run_once(args.dataset, args.method, args.limit, args.warmup, args.workers)

    print("=== batch_serialize Benchmark ===")
    print(f"dataset      : {args.dataset}")
    print(f"method       : {args.method}")
    print(f"warmup       : {args.warmup}")
    print(f"bench_samples: {r['serial']['samples']}")
    print("-- serial --")
    print(f"elapsed_sec  : {r['serial']['elapsed']:.4f}")
    print(f"samples/s    : {r['serial']['samples_s']:.2f}")
    print(f"tokens/s     : {r['serial']['tokens_s']:.2f}")
    print("-- mp (fork) --")
    print(f"workers      : {r['mp']['workers']}")
    print(f"elapsed_sec  : {r['mp']['elapsed']:.4f}")
    print(f"samples/s    : {r['mp']['samples_s']:.2f}")
    print(f"tokens/s     : {r['mp']['tokens_s']:.2f}")
    sp = (r['mp']['samples_s'] / r['serial']['samples_s']) if r['serial']['samples_s'] > 0 else 0.0
    tp = (r['mp']['tokens_s'] / r['serial']['tokens_s']) if r['serial']['tokens_s'] > 0 else 0.0
    print(f"speedup (samples/s): {sp:.2f}x")
    print(f"speedup (tokens/s) : {tp:.2f}x")


if __name__ == "__main__":
    main()

