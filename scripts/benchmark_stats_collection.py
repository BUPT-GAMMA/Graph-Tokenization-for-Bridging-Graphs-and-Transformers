#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基准：统计收集（initialize_with_dataset 内）串行 vs 多进程（fork）

示例：
  python scripts/benchmark_stats_collection.py --dataset qm9test --method feuler --limit 13083 --workers 64

输出：
  - 初始化（含统计收集）耗时对比与加速比
注意：该测试仅对需要统计的序列化器（如 feuler/fcpp）有意义。
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
    ap = argparse.ArgumentParser(description="Benchmark stats collection during serializer initialization")
    ap.add_argument("--dataset", default="qm9test", choices=["qm9", "qm9test", "zinc", "aqsol", "mnist"], help="数据集")
    ap.add_argument("--method", default="feuler", choices=["feuler", "fcpp"], help="需要统计的序列化方法")
    ap.add_argument("--limit", type=int, default=None, help="样本上限（默认全量）")
    ap.add_argument("--workers", type=int, default=64, help="并行进程数（fork）")
    return ap.parse_args()


def take_first_n(data: List[Dict[str, Any]], n: int | None) -> List[Dict[str, Any]]:
    if not n:
        return data
    return data[: min(n, len(data))]


def run_once(dataset: str, method: str, limit: int | None, workers: int):
    cfg = ProjectConfig()
    cfg.dataset.name = dataset

    loader = UnifiedDataFactory.create(cfg.dataset.name, cfg)
    all_graphs, _ = loader.get_all_data_with_indices()
    graphs = take_first_n(all_graphs, limit)

    # 串行统计
    s1 = SerializerFactory.create_serializer(method)
    # 关闭字符串统计，开启/关闭并行
    if hasattr(s1, 'enable_string_stats'):
        s1.enable_string_stats = False
    if hasattr(s1, 'stats_parallel_enabled'):
        s1.stats_parallel_enabled = False
    t0 = time.perf_counter()
    s1.initialize_with_dataset(loader, graph_data_list=graphs)
    dt_serial = time.perf_counter() - t0

    # 多进程统计
    s2 = SerializerFactory.create_serializer(method)
    if hasattr(s2, 'enable_string_stats'):
        s2.enable_string_stats = False
    if hasattr(s2, 'stats_parallel_enabled'):
        s2.stats_parallel_enabled = True
    if hasattr(s2, 'stats_num_workers'):
        s2.stats_num_workers = int(workers)
    t1 = time.perf_counter()
    s2.initialize_with_dataset(loader, graph_data_list=graphs)
    dt_mp = time.perf_counter() - t1

    return dt_serial, dt_mp


def main():
    args = parse_args()
    dt_s, dt_mp = run_once(args.dataset, args.method, args.limit, args.workers)
    speedup = (dt_s / dt_mp) if dt_mp > 0 else 0.0

    print("=== stats collection Benchmark ===")
    print(f"dataset     : {args.dataset}")
    print(f"method      : {args.method}")
    print(f"limit       : {args.limit}")
    print(f"workers     : {args.workers}")
    print(f"serial_sec  : {dt_s:.4f}")
    print(f"mp_sec      : {dt_mp:.4f}")
    print(f"speedup     : {speedup:.2f}x")


if __name__ == "__main__":
    main()

