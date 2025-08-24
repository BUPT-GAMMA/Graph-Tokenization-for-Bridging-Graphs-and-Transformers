#!/usr/bin/env python3
"""
简单并行微调脚本（可CLI配置/适配Slurm）
====================================

与预训练批量脚本保持一致的参数语义：
- 可指定实验组、实验名前缀/标识、数据集/方法/GPU 列表
- BPE 场景仅选择类型（raw/all/random/gaussian），具体区间/均值由下层根据 codebook 自适应
- 微调超参数（单组或JSON多组）
- 任务类型/目标属性、评测聚合模式
- 数据增强（回归任务）：透传到 config.bert.finetuning.regression_augmentation_methods
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
DEFAULT_FT_HYPERPARAMS = [{"epochs": 60, "batch_size": 512, "learning_rate": 1e-4}]
DEFAULT_AGGREGATION_MODE = "best"  # or "best"
DEFAULT_LOG_DIR = "logs/batch_finetune"

# 回归任务增强方法默认集合（与预训练方法名一致，作用于回归任务）
DEFAULT_REG_AUG_METHODS = [
    "random_deletion",
    "random_insertion",
    "random_replacement",
    "random_swap",
    "random_truncation",
]


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
    """移除ANSI颜色控制符，保留中文与Unicode字符。"""
    return _ANSI_RE.sub('', line)

def _stream_to_file(process: subprocess.Popen, file_path: str, sanitize: bool = True) -> None:
    """将子进程输出流写入文件（可选清洗）。"""
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
    # 🆕 支持部分参数指定，缺失的用默认值填充
    if epochs is None and batch_size is None and learning_rate is None:
        return DEFAULT_FT_HYPERPARAMS
    
    # 从默认配置中取基准值
    default_config = DEFAULT_FT_HYPERPARAMS[0]
    final_epochs = epochs if epochs is not None else default_config["epochs"]
    final_batch_size = batch_size if batch_size is not None else default_config["batch_size"]
    final_learning_rate = learning_rate if learning_rate is not None else default_config["learning_rate"]
    
    return [{"epochs": int(final_epochs), "batch_size": int(final_batch_size), "learning_rate": float(final_learning_rate)}]


def create_task_list(datasets: List[str], methods: List[str], bpe_test_configs: List[Dict[str, Any]],
                     hyperparams_list: List[Dict[str, Any]], exp_prefix: str, tag: Optional[str],
                     aug_label: Optional[str], finetune_modes: List[str] = None, 
                     pretrain_exp_prefix: str = "") -> List[Dict[str, Any]]:
    """创建微调任务列表，支持灵活的微调模式选择"""
    tasks: List[Dict[str, Any]] = []
    
    # 默认只使用BERT（向后兼容）
    if finetune_modes is None:
        finetune_modes = ["bert"]
    
    # 根据指定的微调模式构建配置
    encoder_configs = []
    for mode in finetune_modes:
        if mode in {"bert", "bert-pretrain"}:  # bert 作为 bert-pretrain 的别名
            encoder_configs.append({"type": "bert", "": False, "suffix": "", "direct": False, "pretrain_suffix": ""})
        elif mode == "bert-direct":
            encoder_configs.append({"type": "bert", "": False, "suffix": "_bert_direct", "direct": True, "pretrain_suffix": ""})
        elif mode == "gte-direct":
            encoder_configs.append({"type": "Alibaba-NLP/gte-multilingual-base", "": False, "suffix": "_gte_keep_direct", "direct": True, "pretrain_suffix": "_gte_keep"})
        elif mode == "gte-pretrain":
            encoder_configs.append({"type": "Alibaba-NLP/gte-multilingual-base", "": False, "suffix": "_gte_keep_pretrained", "direct": False, "pretrain_suffix": "_gte_keep"})
        elif mode == "gte-reset-direct":
            encoder_configs.append({"type": "Alibaba-NLP/gte-multilingual-base", "": True, "suffix": "_gte_reset_direct", "direct": True, "pretrain_suffix": "_gte_reset"})
        elif mode == "gte-reset-pretrain":
            encoder_configs.append({"type": "Alibaba-NLP/gte-multilingual-base", "": True, "suffix": "_gte_reset_pretrained", "direct": False, "pretrain_suffix": "_gte_reset"})
        else:
            raise ValueError(f"不支持的微调模式: {mode}。支持: bert, bert-pretrain, bert-direct, gte-direct, gte-pretrain, gte-reset-direct, gte-reset-pretrain")
    
    for dataset in datasets:
        for method in methods:
            if method == "smiles" and dataset not in {"qm9", "zinc", "aqsol", "qm9test"}:
                continue
            for bpe_config in bpe_test_configs:
                for encoder_config in encoder_configs:
                    if hyperparams_list:
                        for params in hyperparams_list:
                            bpe_suffix = bpe_config["config_name"]
                            aug_part = f"_{aug_label}" if aug_label else ""
                            encoder_suffix = encoder_config["suffix"]
                            # 删除 _ft_ 和 epoch 标记，确保与预训练阶段的实验名一致
                            exp_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}{encoder_suffix}"
                            experiment_name = f"{exp_prefix}{exp_core}{('_' + tag) if tag else ''}"
                            
                            # 构建预训练实验名（如果需要）
                            pretrain_exp_name = None
                            if not encoder_config["direct"]:  # 需要从预训练模型加载
                                pretrain_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}{encoder_config['pretrain_suffix']}"
                                pretrain_exp_name = f"{pretrain_exp_prefix}{pretrain_core}{('_' + tag) if tag else ''}"
                            
                            tasks.append({
                                "dataset": dataset,
                                "method": method,
                                "hyperparams": params,
                                "bpe_config": bpe_config,
                                "encoder_type": encoder_config["type"],
                                "reset_weights": encoder_config[""],
                                "pretrain_exp_name": pretrain_exp_name,
                                "experiment_name": experiment_name
                            })
                    else:
                        bpe_suffix = bpe_config["config_name"]
                        aug_part = f"_{aug_label}" if aug_label else ""
                        encoder_suffix = encoder_config["suffix"]
                        exp_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}{encoder_suffix}_default"
                        experiment_name = f"{exp_prefix}{exp_core}{('_' + tag) if tag else ''}"
                        
                        # 构建预训练实验名（如果需要）
                        pretrain_exp_name = None
                        if not encoder_config["direct"]:  # 需要从预训练模型加载
                            pretrain_core = f"{dataset}_{method}_{bpe_suffix}{aug_part}{encoder_config['pretrain_suffix']}_default"
                            pretrain_exp_name = f"{pretrain_exp_prefix}{pretrain_core}{('_' + tag) if tag else ''}"
                        
                        tasks.append({
                            "dataset": dataset,
                            "method": method,
                            "hyperparams": None,
                            "bpe_config": bpe_config,
                            "encoder_type": encoder_config["type"],
                            "_weights": encoder_config[""],
                            "pretrain_exp_name": pretrain_exp_name,
                            "experiment_name": experiment_name
                        })
    return tasks


def run_task(task: Dict[str, Any], gpu_id: int, experiment_group: str,
             combined_config_json: Optional[str], aggregation_mode: str,
             log_dir: Optional[str],
             commands_only: bool = False, plain_logs: bool = False,
             commands_file: Optional[str] = None,
             commands_stdout: bool = False,
             save_name_prefix: Optional[str] = None,
             save_name_suffix: Optional[str] = None) -> Optional[subprocess.Popen]:
    cmd = [
        "python", "run_finetune.py",
        "--dataset", task["dataset"],
        "--method", task["method"],
        "--experiment_group", experiment_group,
        "--experiment_name", task["experiment_name"],
        "--device", "auto",
        # "--aggregation_mode", aggregation_mode,
    ]
    if os.environ.get("TG_LOG_STYLE", "").lower() in {"online", "offline"}:
        cmd.extend(["--log_style", os.environ["TG_LOG_STYLE"].lower()])

    # 🆕 收集微调参数 - 任务类型等由数据集自动推断，无需指定
    finetune_extras: list[str] = []
    # finetune_extras.操shouldextend(["--aggregation_mode", aggregation_mode])
    if save_name_prefix:
        finetune_extras.extend(["--save_name_prefix", save_name_prefix])
    if save_name_suffix:
        finetune_extras.extend(["--save_name_suffix", save_name_suffix])

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

    # 🆕 添加编码器相关参数
    if task.get("encoder_type") and task["encoder_type"] != "bert":
        cmd.extend(["--encoder_type", task["encoder_type"]])
    
    if task.get("_weights", False):
        cmd.append("--_weights")
        
    if task.get("pretrain_exp_name"):
        cmd.extend(["--pretrain_exp_name", task["pretrain_exp_name"]])
    
    if combined_config_json:
        cmd.extend(["--config_json", combined_config_json])

    # 将 plain_logs 传递给下层 run_finetune.py，以启用UTF-8与去色包装
    if plain_logs:
        cmd.append("--plain_logs")

    # 目标日志文件（也用于 commands_only 记录）
    stdout_dest = subprocess.PIPE
    log_path = None
    log_fh = None
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"{task['experiment_name']}.log")

    # 末尾追加微调特有参数，确保命令公共部分与预训练一致
    final_cmd = cmd + finetune_extras
    safe_cmd_str = ' '.join(shlex.quote(part) for part in final_cmd)

    if commands_only or commands_stdout:
        record_line = f"CUDA_VISIBLE_DEVICES={gpu_id} {safe_cmd_str}"
        if commands_stdout:
            print(record_line)
        else:
            # 默认写入当前目录 commands.list；如提供 commands_file 则使用之
            dest_file = commands_file or 'commands.list'
            Path(os.path.dirname(dest_file) or '.').mkdir(parents=True, exist_ok=True)
            with open(dest_file, 'a', encoding='utf-8') as fh:
                fh.write(record_line + "\n")
            print(f"✍️ 记录命令到 {dest_file}: {record_line}")
        return None

    if log_path:
        log_fh = open(log_path, 'w', encoding='utf-8', buffering=1)
        stdout_dest = log_fh

    print(f"🚀 GPU {gpu_id}: 开始微调任务 {task['experiment_name']}")
    if log_path:
        print(f"   输出重定向: {log_path}")
    print(f"   命令: {safe_cmd_str}")

    env = os.environ.copy()
    # 尽量避免ANSI与编码问题
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
        # 用管道读取并写入纯净日志
        process = subprocess.Popen(
            final_cmd,
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
            final_cmd,
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
        description="批量并行微调脚本（支持CLI参数覆盖/适配Slurm提交）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 基础与区分参数
    parser.add_argument("--experiment_group", type=str, default=DEFAULT_EXPERIMENT_GROUP, help="实验分组")
    parser.add_argument("--exp_prefix", type=str, default="", help="实验名前缀（会加到自动拼接的名称前）")
    parser.add_argument("--tag", type=str, default=None, help="实验名称附加标识（便于区分批次）")

    # 任务维度
    parser.add_argument("--datasets", type=str, default=','.join(DEFAULT_DATASETS), help="数据集，逗号分隔")
    parser.add_argument("--methods", type=str, default=','.join(DEFAULT_METHODS), help="序列化方法，逗号分隔")
    parser.add_argument("--gpus", type=str, default=','.join(str(x) for x in DEFAULT_GPUS), help="GPU编号，逗号分隔")

    # BPE 场景（仅选择类型，其他参数由下层根据 codebook 自适应）
    parser.add_argument("--bpe_scenarios", type=str, default=','.join(DEFAULT_BPE_SCENARIOS), help="BPE测试场景，逗号分隔: raw,all,random,gaussian（仅选择类型）")

    # 微调超参数（单组）或 JSON 多组
    parser.add_argument("--epochs", type=int, default=None, help="微调轮数（单组超参用）")
    parser.add_argument("--batch_size", type=int, default=None, help="微调批次大小（单组超参用）")
    parser.add_argument("--learning_rate", type=float, default=None, help="微调学习率（单组超参用）")
    parser.add_argument("--hyperparams_json", type=str, default=None, help="多组超参数的JSON（字符串或文件路径），数组形式")

    # 评估配置
    parser.add_argument("--aggregation_mode", type=str, default=DEFAULT_AGGREGATION_MODE, choices=["avg", "best", "learned"], help="测试时增强的聚合模式")

    # 数据增强（回归）
    parser.add_argument("--use_augmentation", type=str, choices=["true", "false"], default=None,
                        help="是否启用回归任务增强（true/false，不指定则保持config默认）")
    
    # 🆕 微调模式选择（灵活配置）
    parser.add_argument("--finetune_modes", type=str, default="bert", 
                        help="要运行的微调模式，逗号分隔。可选: bert,bert-pretrain,bert-direct,gte-direct,gte-pretrain,gte-reset-direct,gte-reset-pretrain。默认bert作为bert-pretrain的别名以保持向后兼容")
    parser.add_argument("--pretrain_exp_prefix", type=str, default="", 
                        help="预训练实验名前缀，用于构建依赖的预训练实验名（仅对*-pretrain模式有效）")

    # JSON覆盖
    parser.add_argument("--config_json", type=str, default=None, help="JSON覆盖（字符串或文件路径）。会与增强开关合并")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="子任务日志目录（每个任务单独一个文件）")
    parser.add_argument("--log_style", type=str, choices=["online", "offline"], default=None, help="日志样式：online=使用tqdm；offline=每个epoch按10%输出摘要")
    parser.add_argument("--commands_only", action="store_true", help="仅记录将要运行的命令到统一文件（append），不实际执行")
    parser.add_argument("--commands_file", type=str, default=None, help="commands_only 模式下的统一命令文件（默认 ./commands.list）")
    parser.add_argument("--commands_stdout", action="store_true", help="仅将将要运行的命令打印到标准输出，不执行也不写文件")
    parser.add_argument("--plain_logs", action="store_true", help="将子任务输出写入无ANSI/emoji的纯文本日志，解决乱码问题")
    parser.add_argument("--save_name_prefix", type=str, default=None, help="仅用于保存目录的实验名前缀（不影响预训练加载）")
    parser.add_argument("--save_name_suffix", type=str, default=None, help="仅用于保存目录的实验名后缀（不影响预训练加载）")

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

    # 仅当用户明确指定增强开关时才覆盖配置（不关心具体方法）
    if args.use_augmentation is not None:
        reg_aug_methods: List[str]
        if args.use_augmentation == "false":
            reg_aug_methods = []
        else:
            reg_aug_methods = list(DEFAULT_REG_AUG_METHODS)

        combined_json_obj = merge_dicts(combined_json_obj, {
            "bert": {
                "finetuning": {
                    "regression_augmentation_methods": reg_aug_methods
                }
            }
        })

    combined_config_json = json.dumps(combined_json_obj, ensure_ascii=False) if combined_json_obj else None

    # 增强标识（纳入实验名）
    aug_label = None
    if args.use_augmentation is not None:
        aug_label = "aug" if args.use_augmentation == "true" else "noaug"

    if not args.commands_stdout:
        print("🚀 开始并行微调...")
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

    if args.log_style:
        combined_json_obj = merge_dicts(combined_json_obj, {"system": {"log_style": args.log_style}})
        combined_config_json = json.dumps(combined_json_obj, ensure_ascii=False)

    # 🆕 解析微调模式
    finetune_modes_list = [mode.strip() for mode in args.finetune_modes.split(',') if mode.strip()] if args.finetune_modes else ["bert"]
    
    tasks = create_task_list(
        datasets=datasets,
        methods=methods,
        bpe_test_configs=bpe_test_configs,
        hyperparams_list=hyperparams_list,
        exp_prefix=args.exp_prefix,
        tag=args.tag,
        aug_label=aug_label,
        finetune_modes=finetune_modes_list,  # 🆕 传递微调模式列表
        pretrain_exp_prefix=args.pretrain_exp_prefix,  # 🆕 传递预训练前缀
    )
    if not args.commands_stdout:
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
            # 启动新任务
            for gpu_id in gpus:
                if gpu_id not in running_processes and task_queue:
                    task = task_queue.pop(0)
                    process = run_task(
                        task,
                        gpu_id,
                        experiment_group=args.experiment_group,
                        combined_config_json=combined_config_json,
                        aggregation_mode=args.aggregation_mode,
                        log_dir=args.log_dir,
                        commands_only=args.commands_only,
                        plain_logs=args.plain_logs,
                        commands_file=args.commands_file,
                        commands_stdout=args.commands_stdout,
                        save_name_prefix=args.save_name_prefix,
                        save_name_suffix=args.save_name_suffix,
                    )
                    if args.commands_only or args.commands_stdout:
                        continue
                    running_processes[gpu_id] = (process, task)
                    # 仅当未重定向日志且未使用纯净日志模式时，才启动流式打印线程
                    if (not hasattr(process, "_tg_log_fh")) and (not args.plain_logs):
                        thread = threading.Thread(target=stream_printer, args=(process, task["experiment_name"], gpu_id))
                        thread.daemon = True
                        thread.start()

            # 检查完成
            completed_gpus = []
            for gpu_id, (process, task) in running_processes.items():
                if process.poll() is not None:
                    return_code = process.wait()
                    if return_code == 0:
                        print(f"✅ GPU {gpu_id}: 微调任务 {task['experiment_name']} 完成")
                        completed_tasks.append(task)
                    else:
                        print(f"❌ GPU {gpu_id}: 微调任务 {task['experiment_name']} 失败 (退出码: {return_code})")
                        failed_tasks.append((task, return_code))
                    if hasattr(process, "_tg_log_fh") and process._tg_log_fh:
                        try:
                            process._tg_log_fh.close()
                        except Exception:
                            pass
                    completed_gpus.append(gpu_id)

            for gpu_id in completed_gpus:
                del running_processes[gpu_id]

            if running_processes and not args.commands_stdout:
                time.sleep(2)

        # 总结
        if not args.commands_stdout:
            print("\n" + "="*60)
            print("📊 微调任务执行总结")
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

            print("\n🎉 所有微调任务完成!")
        return 0 if not failed_tasks else 1

    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
