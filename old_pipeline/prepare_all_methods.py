#!/usr/bin/env python3
"""
批量准备多种序列化方法的缓存工件（序列化 + BPE + 词表）
=====================================================

- 严格遵循“无隐式回退”：缺失即报错；所有构建均为显式调用
- 方法级并发：可通过 --workers 控制并行度
- 词表：默认基于指定方法（或首个方法）的已准备 BPE 序列构建并注册
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple, Dict
from pathlib import Path

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface


def init_worker() -> None:
    # 子进程忽略 Ctrl+C，由主进程统一处理
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def prepare_one(args: Tuple[str, str, str]) -> str:
    """在子进程中独立构建，避免在父进程传递巨大对象/句柄导致 mmap/shm 问题。"""
    dataset, version, method = args
    # 延迟导入，保证子进程上下文干净
    from config import ProjectConfig  # type: ignore
    from src.data.unified_data_interface import UnifiedDataInterface  # type: ignore
    cfg = ProjectConfig()
    cfg.dataset.name = dataset
    udi_local = UnifiedDataInterface(config=cfg, dataset=dataset)
    # 序列化（BPE已集成）
    udi_local.prepare_serialization(method)
    return method


def main():
    parser = argparse.ArgumentParser(description="批量准备多种序列化方法的缓存工件")
    parser.add_argument("--dataset", default="qm9test", help="数据集名称，如 qm9test/zinc，默认 qm9test")
    parser.add_argument("--methods", default=None, help="逗号分隔的方法列表，如 feuler,eulerian,dfs,bfs；未提供则对所有可用方法处理")
    parser.add_argument("--version", default="latest", help="processed/<dataset>/<version> 目录名")
    parser.add_argument("--workers", type=int, default=32, help="方法级并发数量")
    parser.add_argument("--subproc", action="store_true", help="启用子进程按方法并行，避免与内部进程池嵌套冲突")
    parser.add_argument("--child", action="store_true", help="子进程模式：仅执行单一方法的准备工作")

    args = parser.parse_args()

    config = ProjectConfig()
    config.dataset.name = args.dataset

    if args.methods:
        methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        # 未提供 methods：默认使用全部可用序列化方法
        from src.algorithms.serializer.serializer_factory import SerializerFactory
        methods = SerializerFactory.get_available_serializers()
    version = args.version

    # 子进程简化模式：仅执行一个方法后退出
    if args.child:
        try:
            if not args.methods:
                raise ValueError("--child 模式必须通过 --methods 指定且仅指定一个方法")
            ms: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
            if len(ms) != 1:
                raise ValueError("--child 模式仅允许一个方法")
            m = ms[0]
            done = prepare_one((config.dataset.name, version, m))
            print(f"✅ 子进程完成方法: {done}")
            return
        except Exception:
            import traceback as _tb
            print(_tb.format_exc())
            sys.exit(1)

    # 方法级并发：支持子进程或线程池两种模式
    if args.subproc and len(methods) > 1:
        import subprocess
        import threading
        num_methods_workers = max(1, int(args.workers))
        # 构建任务列表
        tasks: List[Tuple[str, List[str]]] = []
        for m in methods:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset", args.dataset,
                "--methods", m,
                "--version", version,
                "--workers", "1",
                "--child",
            ]
            tasks.append((m, cmd))

        active: Dict[str, Tuple[subprocess.Popen, threading.Thread]] = {}
        pending = list(tasks)

        def _pump_stdout(proc: subprocess.Popen, method_key: str):
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    print(f"[{method_key}] {line}")
            except Exception:
                import traceback
                print(f"[{method_key}] 输出读取异常:\n{traceback.format_exc()}")

        def _start_next():
            if not pending:
                return
            m, cmd = pending.pop(0)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            t = threading.Thread(target=_pump_stdout, args=(proc, m), daemon=True)
            t.start()
            active[m] = (proc, t)

        for _ in range(min(num_methods_workers, len(pending))):
            _start_next()

        while active:
            to_remove = []
            for m, (proc, t) in list(active.items()):
                rc = proc.poll()
                if rc is not None:
                    t.join(timeout=1)
                    if rc != 0:
                        print(f"❌ 子进程失败: {m} (rc={rc})")
                    else:
                        print(f"✅ 子进程完成: {m}")
                    to_remove.append(m)
            for m in to_remove:
                active.pop(m, None)
                _start_next()
            import time as _time
            _time.sleep(0.05)
    else:
        # 线程池并发：避免与内部进程池嵌套冲突
        from concurrent.futures import ThreadPoolExecutor, as_completed
        num_methods_workers = max(1, int(args.workers))
        tasks_futs = []
        with ThreadPoolExecutor(max_workers=num_methods_workers) as ex:
            for m in methods:
                tasks_futs.append(ex.submit(prepare_one, (config.dataset.name, version, m)))
            for fut in as_completed(tasks_futs):
                m_done = fut.result()
                print(f"✅ 已完成方法: {m_done}")

    # 注意：词表构建已移除，需要使用其他工具显式构建词表
    print("\n📚 词表构建已从此脚本移除")
    print("💡 提示：如需词表，请使用独立的词表构建工具或在训练时动态构建")


if __name__ == "__main__":
    main()


