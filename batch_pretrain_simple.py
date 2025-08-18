#!/usr/bin/env python3
"""
简单并行预训练脚本（可CLI配置/适配Slurm）
=====================================

在多GPU上并行运行预训练任务；支持命令行配置实验名前缀、增强开关、BPE场景与超参数，便于在集群平台提交与区分实验。
"""

import subprocess
import time
import threading
from typing import List, Dict, Any, Optional
import os
import signal
import sys
import argparse
import json
from pathlib import Path
import re
import shlex


# ===== 默认配置（可被CLI覆盖） =====
DEFAULT_EXPERIMENT_GROUP = "test_zinc_10multi_4bpe-way"
DEFAULT_DATASETS = ["zinc"]
DEFAULT_METHODS = ["feuler", "eulerian", "cpp", "fcpp", "topo", "smiles"]
DEFAULT_GPUS = [0]
# DEFAULT_BPE_SCENARIOS = ["raw", "all", "random", "gaussian"]
DEFAULT_BPE_SCENARIOS = ["all"]
DEFAULT_HYPERPARAMS = [{"epochs": 100, "batch_size": 1024, "learning_rate": 5e-4}]
DEFAULT_MLM_AUG_METHODS = [
    "random_deletion",
    "random_insertion",
    "random_replacement",
    "random_swap",
    "random_truncation",
]
DEFAULT_LOG_DIR = "logs/batch_pretrain"


def parse_list_arg(value: str) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_int_list_arg(value: str) -> List[int]:
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


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def _sanitize_line(line: str) -> str:
    return _ANSI_RE.sub('', line)

def _stream_to_file(process: subprocess.Popen, file_path: str, sanitize: bool = True) -> None:
    with open(file_path, 'a', encoding='utf-8') as fh:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            fh.write(_sanitize_line(line) if sanitize else line)
            fh.flush()

def build_bpe_test_configs(scenarios: List[str]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for sc in scenarios:
        if sc == "raw":
            configs.append({"config_name": "raw", "bpe_encode_rank_mode": "none"})
        elif sc == "all":
            configs.append({"config_name": "all", "bpe_encode_rank_mode": "all"})
        elif sc == "random":
            configs.append({"config_name": "random", "bpe_encode_rank_mode": "random"})
        elif sc == "gaussian":
            configs.append({"config_name": "gaussian", "bpe_encode_rank_mode": "gaussian"})
        else:
            raise ValueError(f"不支持的BPE场景: {sc}")
    return configs


def build_hyperparams_list(hp_json: Optional[str], epochs: Optional[int],
                           batch_size: Optional[int], learning_rate: Optional[float]) -> List[Dict[str, Any]]:
    if hp_json:
        loaded = load_json_input(hp_json)
        if not isinstance(loaded, list):
            raise ValueError("--hyperparams_json 必须是包含若干对象的JSON数组")
        for item in loaded:
            if not isinstance(item, dict) or not {"epochs", "batch_size", "learning_rate"} <= set(item.keys()):
                raise ValueError("--hyperparams_json 中每个对象必须包含 epochs, batch_size, learning_rate 三个键")
        return loaded
    if epochs is None and batch_size is None and learning_rate is None:
        return DEFAULT_HYPERPARAMS
    if epochs is None or batch_size is None or learning_rate is None:
        raise ValueError("使用独立参数指定超参数时，必须同时提供 --epochs, --batch_size, --learning_rate")
    return [{"epochs": int(epochs), "batch_size": int(batch_size), "learning_rate": float(learning_rate)}]


def create_task_list(datasets: List[str], methods: List[str], bpe_test_configs: List[Dict[str, Any]],
                     hyperparams_list: List[Dict[str, Any]], exp_prefix: str, tag: Optional[str],
                     aug_label: Optional[str]) -> List[Dict[str, Any]]:
    """创建任务列表"""
    tasks: List[Dict[str, Any]] = []
    for dataset in datasets:
        for method in methods:
            if method == "smiles" and dataset not in {"qm9", "zinc", "aqsol", "qm9test"}:
                continue
            for bpe_config in bpe_test_configs:
                if hyperparams_list:
                    for params in hyperparams_list:
                        bpe_suffix = bpe_config["config_name"]
                        aug_part = f"_{aug_label}" if aug_label else ""
                        # 删除 epoch 标记，确保与微调阶段的实验名一致
                        exp_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}"
                        experiment_name = f"{exp_prefix}{exp_core}{('_' + tag) if tag else ''}"
                        tasks.append({
                            "dataset": dataset,
                            "method": method,
                            "hyperparams": params,
                            "bpe_config": bpe_config,
                            "experiment_name": experiment_name
                        })
                else:
                    bpe_suffix = bpe_config["config_name"]
                    aug_part = f"_{aug_label}" if aug_label else ""
                    exp_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}_default"
                    experiment_name = f"{exp_prefix}{exp_core}{('_' + tag) if tag else ''}"
                    tasks.append({
                        "dataset": dataset,
                        "method": method,
                        "hyperparams": None,
                        "bpe_config": bpe_config,
                        "experiment_name": experiment_name
                    })
    return tasks


def run_task(task: Dict[str, Any], gpu_id: int, experiment_group: str,
             combined_config_json: Optional[str], log_dir: Optional[str],
             commands_only: bool = False, plain_logs: bool = False,
             commands_file: Optional[str] = None) -> Optional[subprocess.Popen]:
    """在指定GPU上运行单个任务"""
    cmd = [
        "python", "run_pretrain.py",
        "--dataset", task["dataset"],
        "--method", task["method"],
        "--experiment_group", experiment_group,
        "--experiment_name", task["experiment_name"],
        "--device", "auto"
    ]
    # 透传离线日志样式：批量脚本通常倾向 offline 以减少tqdm
    if os.environ.get("TG_LOG_STYLE", "").lower() in {"online", "offline"}:
        cmd.extend(["--log_style", os.environ["TG_LOG_STYLE"].lower()])

    bpe_config = task["bpe_config"]
    if "bpe_encode_rank_mode" in bpe_config and bpe_config["bpe_encode_rank_mode"]:
        cmd.extend(["--bpe_encode_rank_mode", str(bpe_config["bpe_encode_rank_mode"])])

    if task["hyperparams"]:
        params = task["hyperparams"]
        cmd.extend([
            "--epochs", str(params["epochs"]),
            "--batch_size", str(params["batch_size"]),
            "--learning_rate", str(params["learning_rate"])
        ])

    if combined_config_json:
        cmd.extend(["--config_json", combined_config_json])

    # 将 plain_logs 传递给下层 run_pretrain.py，以启用UTF-8与去色包装
    if plain_logs:
        cmd.append("--plain_logs")

    # 目标日志文件（也用于 commands_only 记录）
    stdout_dest = subprocess.PIPE
    log_path = None
    log_fh = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"{task['experiment_name']}.log")

    safe_cmd_str = ' '.join(shlex.quote(part) for part in cmd)

    if commands_only:
        # 只记录命令（包含 CUDA_VISIBLE_DEVICES 环境）到统一文件（追加）
        record_line = f"CUDA_VISIBLE_DEVICES={gpu_id} {safe_cmd_str}\n"
        # 默认写入当前目录 commands.list；如提供 commands_file 则使用之
        dest_file = commands_file or 'commands.list'
        Path(os.path.dirname(dest_file) or '.').mkdir(parents=True, exist_ok=True)
        with open(dest_file, 'a', encoding='utf-8') as fh:
            fh.write(record_line)
        print(f"✍️ 记录命令到 {dest_file}: {record_line.strip()}")
        return None

    # 执行模式：重定向子进程输出
    if log_path:
        log_fh = open(log_path, 'w', encoding='utf-8', buffering=1)
        stdout_dest = log_fh

    print(f"🚀 GPU {gpu_id}: 开始任务 {task['experiment_name']}")
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

    if plain_logs and log_path:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        t = threading.Thread(target=_stream_to_file, args=(process, log_path, True))
        t.daemon = True
        t.start()
    else:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_dest,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

    # 保存日志句柄到进程对象，便于完成后关闭
    if log_fh is not None:
        process._tg_log_fh = log_fh  # type: ignore[attr-defined]
        process._tg_log_path = log_path  # type: ignore[attr-defined]

    return process


def stream_printer(process: subprocess.Popen, task_name: str, gpu_id: int):
    """多线程打印机，读取并打印一个子进程的输出流"""
    for line in iter(process.stdout.readline, ''):
        if line:
            print(f"[GPU{gpu_id}-{task_name}] {line.strip()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量并行预训练脚本（支持CLI参数覆盖/适配Slurm提交）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--experiment_group", type=str, default=DEFAULT_EXPERIMENT_GROUP, help="实验分组")
    parser.add_argument("--exp_prefix", type=str, default="", help="实验名前缀（会加到自动拼接的名称前）")
    parser.add_argument("--tag", type=str, default=None, help="实验名称附加标识（便于区分批次）")

    parser.add_argument("--datasets", type=str, default=','.join(DEFAULT_DATASETS), help="数据集，逗号分隔")
    parser.add_argument("--methods", type=str, default=','.join(DEFAULT_METHODS), help="序列化方法，逗号分隔")
    parser.add_argument("--gpus", type=str, default=','.join(str(x) for x in DEFAULT_GPUS), help="GPU编号，逗号分隔")

    parser.add_argument("--bpe_scenarios", type=str, default=','.join(DEFAULT_BPE_SCENARIOS), help="BPE测试场景，逗号分隔: raw,all,random,gaussian（仅选择类型，下层根据codebook自适应范围）")

    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（单组超参用）")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小（单组超参用）")
    parser.add_argument("--learning_rate", type=float, default=None, help="学习率（单组超参用）")
    parser.add_argument("--hyperparams_json", type=str, default=None, help="多组超参数的JSON（字符串或文件路径），数组形式")

    parser.add_argument("--use_augmentation", type=str, choices=["true", "false"], default=None,
                        help="是否启用MLM增强（true/false，不指定则保持config默认）")

    parser.add_argument("--config_json", type=str, default=None,
                        help="JSON覆盖（字符串或文件路径）。会与增强开关产生的覆盖合并")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="子任务日志目录（每个任务单独一个文件）")
    parser.add_argument("--log_style", type=str, choices=["online", "offline"], default=None, help="日志样式：online=使用tqdm；offline=每个epoch按10%输出摘要")
    parser.add_argument("--commands_only", action="store_true", help="仅记录将要运行的命令到统一文件（append），不实际执行")
    parser.add_argument("--commands_file", type=str, default=None, help="commands_only 模式下的统一命令文件；未指定则写入 <log_dir>/commands.list")
    parser.add_argument("--plain_logs", action="store_true", help="将子任务输出写入无ANSI/emoji的纯文本日志，解决乱码问题")

    args = parser.parse_args()

    datasets = parse_list_arg(args.datasets) or DEFAULT_DATASETS
    methods = parse_list_arg(args.methods) or DEFAULT_METHODS
    gpus = parse_int_list_arg(args.gpus) or DEFAULT_GPUS

    scenarios = parse_list_arg(args.bpe_scenarios) or DEFAULT_BPE_SCENARIOS
    bpe_test_configs = build_bpe_test_configs(scenarios)

    hyperparams_list = build_hyperparams_list(
        hp_json=args.hyperparams_json,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    combined_json_obj: Dict[str, Any] = {}
    if args.config_json:
        loaded = load_json_input(args.config_json)
        combined_json_obj = merge_dicts(combined_json_obj, loaded)

    if args.use_augmentation is not None:
        aug_methods: List[str]
        if args.use_augmentation == "false":
            aug_methods = []
        else:
            aug_methods = list(DEFAULT_MLM_AUG_METHODS)
        combined_json_obj = merge_dicts(combined_json_obj, {
            "bert": {
                "pretraining": {
                    "mlm_augmentation_methods": aug_methods
                }
            }
        })

    combined_config_json = json.dumps(combined_json_obj, ensure_ascii=False) if combined_json_obj else None

    # 构造增强标识用于实验名（仅当用户显式指定时）
    aug_label = None
    if args.use_augmentation is not None:
        aug_label = "aug" if args.use_augmentation == "true" else "noaug"

    print("🚀 开始并行预训练...")
    print(f"实验组: {args.experiment_group}")
    print(f"数据集: {datasets}")
    print(f"方法: {methods}")
    print(f"BPE场景: {scenarios}")
    print(f"可用GPU: {gpus}")
    if combined_config_json:
        print("📝 合并后的JSON覆盖将传入子进程 --config_json")
    if args.exp_prefix:
        print(f"🏷️ 实验名前缀: {args.exp_prefix}")
    if args.tag:
        print(f"🏷️ 实验名附加标识: {args.tag}")

    # 如果提供了 --log_style，则以 JSON 覆盖传递给子进程（与直接 --log_style 二选一都可生效）
    if args.log_style:
        combined_json_obj = merge_dicts(combined_json_obj, {"system": {"log_style": args.log_style}})
        combined_config_json = json.dumps(combined_json_obj, ensure_ascii=False)

    tasks = create_task_list(
        datasets=datasets,
        methods=methods,
        bpe_test_configs=bpe_test_configs,
        hyperparams_list=hyperparams_list,
        exp_prefix=args.exp_prefix,
        tag=args.tag,
        aug_label=aug_label,
    )
    print(f"总任务数: {len(tasks)}")
    if args.commands_only:
        target_file = 'commands.list'
        print(f"✍️ commands_only 模式：仅记录命令到 {target_file}，不执行子任务")

    running_processes = {}
    task_queue = tasks.copy()
    completed_tasks: List[Dict[str, Any]] = []
    failed_tasks: List[tuple] = []

    def signal_handler(sig, frame):
        print("\n⚠️ 收到中断信号，正在终止所有任务...")
        for gpu_id, (process, task) in running_processes.items():
            if process.poll() is None:
                print(f"🛑 终止 GPU {gpu_id} 上的任务: {task['experiment_name']}")
                process.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while task_queue or running_processes:
            for gpu_id in gpus:
                if gpu_id not in running_processes and task_queue:
                    task = task_queue.pop(0)
                    process = run_task(task, gpu_id, args.experiment_group, combined_config_json, args.log_dir, args.commands_only, args.plain_logs, args.commands_file)
                    if args.commands_only:
                        continue
                    running_processes[gpu_id] = (process, task)
                    if (not hasattr(process, "_tg_log_fh")) and (not args.plain_logs):
                        thread = threading.Thread(target=stream_printer, args=(process, task["experiment_name"], gpu_id))
                        thread.daemon = True
                        thread.start()

            completed_gpus = []
            for gpu_id, (process, task) in running_processes.items():
                if process.poll() is not None:
                    return_code = process.wait()
                    if return_code == 0:
                        print(f"✅ GPU {gpu_id}: 任务 {task['experiment_name']} 完成")
                        completed_tasks.append(task)
                    else:
                        print(f"❌ GPU {gpu_id}: 任务 {task['experiment_name']} 失败 (退出码: {return_code})")
                        failed_tasks.append((task, return_code))
                    # 关闭日志文件句柄
                    if hasattr(process, "_tg_log_fh") and process._tg_log_fh:
                        try:
                            process._tg_log_fh.close()
                        except Exception:
                            pass
                    completed_gpus.append(gpu_id)

            for gpu_id in completed_gpus:
                del running_processes[gpu_id]

            if running_processes:
                time.sleep(2)

        print("\n" + "="*60)
        print("📊 任务执行总结")
        print("="*60)
        print(f"✅ 成功完成: {len(completed_tasks)}")
        print(f"❌ 执行失败: {len(failed_tasks)}")

        if completed_tasks:
            print("\n✅ 成功任务:")
            for task in completed_tasks:
                print(f"  - {task['experiment_name']}")

        if failed_tasks:
            print("\n❌ 失败任务:")
            for task, code in failed_tasks:
                print(f"  - {task['experiment_name']} (退出码: {code})")

        print("\n🎉 所有任务完成!")
        return 0 if not failed_tasks else 1

    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
