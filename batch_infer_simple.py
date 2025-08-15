#!/usr/bin/env python3
"""
简单并行推理脚本（与 batch_finetune_simple 对等）
===============================================

- 从命令行指定多个数据集/方法/GPU，批量运行已微调模型的推理
- 与微调保持对等的参数语义与日志结构
- 支持 learned 聚合（在推理阶段训练聚合器）
"""

from __future__ import annotations

import subprocess
import threading
from typing import List, Dict, Any, Optional
import os
import sys
import argparse
import json
from pathlib import Path
import re
import shlex


DEFAULT_EXPERIMENT_GROUP = "infer_group"
DEFAULT_DATASETS = ["qm9"]
DEFAULT_METHODS = ["feuler"]
DEFAULT_GPUS = [0]
DEFAULT_AGGREGATION_MODE = "avg"  # 可选: avg|best|learned
DEFAULT_LOG_DIR = "logs/batch_infer"

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _sanitize_line(line: str) -> str:
    return _ANSI_RE.sub('', line)


def parse_list_arg(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_int_list_arg(value: Optional[str]) -> List[int]:
    if value is None or value.strip() == "":
        return []
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def load_json_input(json_input: str) -> Any:
    text = json_input.strip()
    if text.startswith('{') or text.startswith('['):
        return json.loads(json_input)
    path = Path(json_input)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {json_input}")
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def run_task(task: Dict[str, Any], gpu_id: int, experiment_group: str,
             log_dir: Optional[str] = None, plain_logs: bool = False,
             commands_only: bool = False, commands_file: Optional[str] = None) -> Optional[subprocess.Popen]:
    cmd = [
        "python", "run_infer.py",
        "--dataset", task["dataset"],
        "--method", task["method"],
        "--task", task["task"],
        "--aggregation_mode", task["aggregation_mode"],
        "--experiment_group", experiment_group,
    ]
    if task.get("num_classes") is not None:
        cmd.extend(["--num_classes", str(task["num_classes"])])
    if task.get("model_dir"):
        cmd.extend(["--model_dir", task["model_dir"]])
    if task.get("save_name"):
        cmd.extend(["--save_name", task["save_name"]])
    if task.get("save_name_suffix"):
        cmd.extend(["--save_name_suffix", task["save_name_suffix"]])
    if task.get("log_style"):
        cmd.extend(["--log_style", task["log_style"]])

    # 记录/输出
    stdout_dest = subprocess.PIPE
    log_path = None
    log_fh = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"{task['dataset']}_{task['method']}_{task['task']}.log")

    safe_cmd_str = ' '.join(shlex.quote(part) for part in cmd)

    if commands_only:
        record_line = f"CUDA_VISIBLE_DEVICES={gpu_id} {safe_cmd_str}\n"
        dest_file = commands_file or 'commands_infer.list'
        Path(os.path.dirname(dest_file) or '.').mkdir(parents=True, exist_ok=True)
        with open(dest_file, 'a', encoding='utf-8') as fh:
            fh.write(record_line)
        print(f"✍️ 记录命令到 {dest_file}: {record_line.strip()}")
        return None

    if log_path:
        log_fh = open(log_path, 'w', encoding='utf-8', buffering=1)
        stdout_dest = log_fh

    print(f"🔎 GPU {gpu_id}: 推理 {task['dataset']}/{task['method']} ({task['task']})")
    if log_path:
        print(f"   输出重定向: {log_path}")
    print(f"   命令: {safe_cmd_str}")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    if plain_logs:
        env["NO_COLOR"] = "1"
        env["CLICOLOR"] = "0"
        env["FORCE_COLOR"] = "0"
        env["TERM"] = "dumb"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    process = subprocess.Popen(
        cmd,
        stdout=stdout_dest,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    if log_fh is not None:
        process._tg_log_fh = log_fh  # type: ignore[attr-defined]
        process._tg_log_path = log_path  # type: ignore[attr-defined]

    return process


def stream_printer(process: subprocess.Popen, task_name: str, gpu_id: int):
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[GPU{gpu_id}-{task_name}] {line.strip()}")


def main():
    parser = argparse.ArgumentParser(
        description="批量并行推理脚本（与微调对等）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment_group", type=str, default=DEFAULT_EXPERIMENT_GROUP, help="实验分组")
    parser.add_argument("--datasets", type=str, default=','.join(DEFAULT_DATASETS), help="数据集列表，逗号分隔")
    parser.add_argument("--methods", type=str, default=','.join(DEFAULT_METHODS), help="方法列表，逗号分隔")
    parser.add_argument("--gpus", type=str, default=','.join(str(x) for x in DEFAULT_GPUS), help="GPU编号，逗号分隔")
    parser.add_argument("--aggregation_mode", type=str, default=DEFAULT_AGGREGATION_MODE, choices=["avg", "best", "learned"], help="聚合模式")
    parser.add_argument("--task", type=str, choices=["regression", "classification"], required=True, help="任务类型")
    parser.add_argument("--num_classes", type=int, default=None, help="分类类别数（分类任务需要或自动推断）")
    parser.add_argument("--model_dir", type=str, default=None, help="显式指定模型目录（可选）")
    parser.add_argument("--save_name", type=str, default="finetune", help="微调保存子目录名（默认为 finetune，用于自动解析")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="子任务日志目录")
    parser.add_argument("--log_style", type=str, choices=["online", "offline"], default=None, help="日志样式")
    parser.add_argument("--commands_only", action="store_true", help="仅记录命令不执行")
    parser.add_argument("--commands_file", type=str, default=None, help="commands_only 模式下的命令输出文件")
    parser.add_argument("--plain_logs", action="store_true", help="输出去色，纯文本日志")

    args = parser.parse_args()

    datasets = parse_list_arg(args.datasets) or DEFAULT_DATASETS
    methods = parse_list_arg(args.methods) or DEFAULT_METHODS
    gpus = parse_int_list_arg(args.gpus) or DEFAULT_GPUS

    tasks: List[Dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            tasks.append({
                "dataset": dataset,
                "method": method,
                "task": args.task,
                "aggregation_mode": args.aggregation_mode,
                "num_classes": args.num_classes,
                "model_dir": args.model_dir,
                "save_name": args.save_name,
                "log_style": args.log_style,
                # 透传微调常用参数，保证与微调等价
                "save_name_prefix": None,
                "save_name_suffix": args.save_name_suffix if hasattr(args, 'save_name_suffix') else None,
            })

    print("🚀 开始并行推理...")
    print(f"实验组: {args.experiment_group}")
    print(f"数据集: {datasets}")
    print(f"方法: {methods}")
    print(f"可用GPU: {gpus}")
    print(f"聚合: {args.aggregation_mode}")

    processes: List[subprocess.Popen] = []
    gpu_ptr = 0
    for idx, task in enumerate(tasks):
        gpu_id = gpus[gpu_ptr % len(gpus)]
        gpu_ptr += 1
        p = run_task(task, gpu_id, args.experiment_group, log_dir=args.log_dir,
                     plain_logs=args.plain_logs, commands_only=args.commands_only,
                     commands_file=args.commands_file)
        if p is not None:
            processes.append(p)

    # 等待所有子任务
    for p in processes:
        p.wait()
        if hasattr(p, '_tg_log_fh') and getattr(p, '_tg_log_fh') is not None:
            try:
                getattr(p, '_tg_log_fh').close()
            except Exception:
                pass

    print("✅ 并行推理完成")

    return 0


if __name__ == "__main__":
    sys.exit(main())


