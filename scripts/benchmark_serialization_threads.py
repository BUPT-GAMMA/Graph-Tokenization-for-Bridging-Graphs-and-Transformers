#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基准脚本：比较序列化在默认线程与强制单线程模式下的吞吐

用法示例：
  1) 单方法基准（默认线程）
     python scripts/benchmark_serialization_threads.py --dataset qm9test --method feuler --limit 1000 --warmup 20

  2) 单方法基准（强制单线程）
     python scripts/benchmark_serialization_threads.py --dataset qm9test --method feuler --limit 1000 --warmup 20 --single_thread

  3) 多方法+画图（同时跑默认与单线程，对比输出CSV与PNG）
     python scripts/benchmark_serialization_threads.py --dataset qm9test --methods feuler,cpp,fcpp --limit 1000 --warmup 20 --plot --out_dir log/bench_serial

说明：
  - 使用统一配置 `config.ProjectConfig()`；不做隐式回退；如缺文件会抛错。
  - `feuler` 初始化需要统计信息，脚本直接传入全部数据列表。
  - 统计指标：样本/秒、token/秒；其中 token 统计为拼接后的 token 数量（首个序列）。
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path

# 确保项目根目录在 sys.path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# 注意：如需强制单线程，必须在导入 torch/dgl 之前设置环境变量

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serialization threading benchmark")
    parser.add_argument("--dataset", type=str, default="qm9test", choices=["qm9", "qm9test", "zinc", "aqsol", "mnist"], help="数据集名称")
    parser.add_argument("--method", type=str, default=None, help="单方法名称（若提供 --methods 则忽略该项）")
    parser.add_argument("--methods", type=str, default=None, help="多方法，逗号分隔，如 feuler,cpp,fcpp")
    parser.add_argument("--limit", type=int, default=1000, help="参与基准的样本数上限（按数据顺序截取）")
    parser.add_argument("--warmup", type=int, default=20, help="预热样本数（不计入统计）")
    parser.add_argument("--single_thread", action="store_true", help="强制单线程（OMP/MKL/OPENBLAS/TBB/NUMEXPR/DGL/torch=1）")
    parser.add_argument("--plot", action="store_true", help="多方法模式下，生成CSV与PNG")
    parser.add_argument("--out_dir", type=str, default="log/bench_serial", help="结果输出目录（CSV/PNG）")
    return parser.parse_args()


def configure_single_thread_env() -> None:
    # 设置常见数值后端线程数为 1
    env_map = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TBB_NUM_THREADS": "1",
        "DGL_NUM_THREADS": "1",
    }
    for k, v in env_map.items():
        os.environ[k] = v


def import_runtime_modules(single_thread: bool):
    if single_thread:
        configure_single_thread_env()
    # 再导入 torch/dgl 以及项目模块
    import torch  # noqa: F401
    from config import ProjectConfig  # noqa: E402
    from src.data.unified_data_factory import UnifiedDataFactory  # noqa: E402
    from src.algorithms.serializer.serializer_factory import SerializerFactory  # noqa: E402
    return ProjectConfig, UnifiedDataFactory, SerializerFactory


def summarize_threads(single_thread: bool) -> Dict[str, Any]:
    import torch
    keys = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS", "DGL_NUM_THREADS"
    ]
    env = {k: os.environ.get(k, "") for k in keys}
    return {
        "single_thread": bool(single_thread),
        "torch_num_threads": int(getattr(torch, "get_num_threads", lambda: 0)()),
        "torch_num_interop_threads": int(getattr(torch, "get_num_interop_threads", lambda: 0)()),
        "env": env,
    }


def force_torch_single_thread():
    import torch
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


def take_first_n(data: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return []
    return data[: min(n, len(data))]


def build_serializer(method: str, dataset_loader, all_graphs: List[Dict[str, Any]]):
    from src.algorithms.serializer.serializer_factory import SerializerFactory
    serializer = SerializerFactory.create_serializer(method)
    # feuler/fcpp 需要统计，直接传全部图；其他方法可忽略
    try:
        serializer.initialize_with_dataset(dataset_loader, graph_data_list=all_graphs)
    except TypeError:
        serializer.initialize_with_dataset(dataset_loader)
    return serializer


def _serialize_and_count_tokens(serializer, sample: Dict[str, Any]) -> int:
    res = serializer.serialize(sample)
    ts, _ = res.get_sequence(0)
    return int(len(ts))


def run_once(method: str, dataset: str, limit: int, warmup: int, single_thread: bool) -> Tuple[float, int, int]:
    ProjectConfig, UnifiedDataFactory, SerializerFactory = import_runtime_modules(single_thread)

    # torch 线程设置（导入后可用）；如需单线程则强制 1
    if single_thread:
        force_torch_single_thread()

    cfg = ProjectConfig()
    cfg.dataset.name = dataset

    loader = UnifiedDataFactory.create(cfg.dataset.name, cfg)
    all_graphs, _ = loader.get_all_data_with_indices()

    serializer = build_serializer(method, loader, all_graphs)

    warm_list = take_first_n(all_graphs, warmup)
    bench_list = take_first_n(all_graphs[warmup:], limit)

    for sample in warm_list:
        _ = _serialize_and_count_tokens(serializer, sample)

    t0 = time.perf_counter()
    total_tokens = 0
    total_ok = 0
    for sample in bench_list:
        tokens = _serialize_and_count_tokens(serializer, sample)
        total_tokens += tokens
        total_ok += 1
    dt = time.perf_counter() - t0

    return dt, total_ok, total_tokens


def run_single_method_cli(dataset: str, method: str, limit: int, warmup: int, single_thread: bool) -> None:
    dt, total_ok, total_tokens = run_once(method, dataset, limit, warmup, single_thread)
    threads_info = summarize_threads(single_thread)
    print("=== Serialization Benchmark ===")
    print(f"dataset            : {dataset}")
    print(f"method             : {method}")
    print(f"warmup             : {warmup}")
    print(f"bench_samples      : {total_ok}")
    print(f"elapsed_sec        : {dt:.4f}")
    print(f"throughput_samples : {total_ok / dt if dt > 0 else 0:.2f} samples/s")
    print(f"total_tokens       : {total_tokens}")
    print(f"throughput_tokens  : {total_tokens / dt if dt > 0 else 0:.2f} tokens/s")
    print(f"single_thread      : {threads_info['single_thread']}")
    print(f"torch_threads      : {threads_info['torch_num_threads']} / interop {threads_info['torch_num_interop_threads']}")
    print("env_threads:")
    for k, v in threads_info["env"].items():
        print(f"  - {k}={v}")


def run_multi_methods_plot(dataset: str, methods: List[str], limit: int, warmup: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ["method,mode,elapsed_sec,throughput_samples,throughput_tokens"]

    import subprocess
    import re

    def _run_sub(method: str, single: bool) -> Tuple[float, float, float]:
        cmd = [
            sys.executable,
            str(PROJ_ROOT / "scripts" / "benchmark_serialization_threads.py"),
            "--dataset", dataset,
            "--method", method,
            "--limit", str(limit),
            "--warmup", str(warmup),
        ]
        if single:
            cmd.append("--single_thread")
        env = os.environ.copy()
        # 不在父进程内更改任何线程相关设置，全部交给子进程
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(PROJ_ROOT), text=True)
        out = p.stdout
        if p.returncode != 0:
            print(out)
            raise RuntimeError(f"subprocess failed for {method} single={single}")
        # 解析
        thr_s = thr_t = elapsed = 0.0
        for line in out.splitlines():
            m1 = re.search(r"throughput_samples\s*:\s*([0-9.]+)", line)
            m2 = re.search(r"throughput_tokens\s*:\s*([0-9.]+)", line)
            m3 = re.search(r"elapsed_sec\s*:\s*([0-9.]+)", line)
            if m1:
                thr_s = float(m1.group(1))
            if m2:
                thr_t = float(m2.group(1))
            if m3:
                elapsed = float(m3.group(1))
        return elapsed, thr_s, thr_t

    results = {}
    for mode in [False, True]:  # False=默认线程, True=单线程
        for m in methods:
            elapsed, thr_s, thr_t = _run_sub(m, mode)
            rows.append(f"{m},{'single' if mode else 'default'},{elapsed:.6f},{thr_s:.6f},{thr_t:.6f}")
            results.setdefault(m, {})['single' if mode else 'default'] = (thr_s, thr_t)
            print(f"[{m}][{'single' if mode else 'default'}] samples/s={thr_s:.2f}, tokens/s={thr_t:.2f}")

    csv_path = out_dir / f"bench_{dataset}_methods_threads.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")

    # 画图
    import matplotlib.pyplot as plt
    labels = methods
    default_vals = [results[m]['default'][0] for m in methods]
    single_vals = [results[m]['single'][0] for m in methods]
    x = range(len(labels))

    plt.figure(figsize=(8, 4))
    w = 0.35
    plt.bar([i - w/2 for i in x], default_vals, width=w, label='default-threads')
    plt.bar([i + w/2 for i in x], single_vals, width=w, label='single-thread')
    plt.xticks(list(x), labels)
    plt.ylabel('samples/s')
    plt.title(f'{dataset}: serialization throughput (limit={limit}, warmup={warmup})')
    plt.legend()
    plt.tight_layout()

    png_path = out_dir / f"bench_{dataset}_methods_threads.png"
    plt.savefig(png_path, dpi=150)
    print(f"CSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',') if len(m.strip()) > 0]
        assert len(methods) > 0, "methods 不能为空"
        assert args.plot, "多方法模式请加 --plot 以输出CSV/PNG"
        run_multi_methods_plot(dataset=args.dataset,
                               methods=methods,
                               limit=args.limit,
                               warmup=args.warmup,
                               out_dir=Path(args.out_dir))
    else:
        assert args.method is not None, "单方法模式需要 --method"
        run_single_method_cli(dataset=args.dataset,
                              method=args.method,
                              limit=args.limit,
                              warmup=args.warmup,
                              single_thread=args.single_thread)
