#!/usr/bin/env python3
"""
超参数搜索脚本
==============

在多GPU上并行进行超参数搜索，支持预训练和微调的超参数搜索。
"""

import subprocess
import time
import json
import itertools
from typing import List, Dict, Any, Tuple
import os
import signal
import sys
from pathlib import Path


# ===== 配置区域 =====
EXPERIMENT_GROUP = "hyperparam_search"
DATASET = "qm9test"
GPUS = [0, 1]  # 可用的GPU编号

# 搜索任务类型
SEARCH_TYPE = "pretrain"  # "pretrain" 或 "finetune"

# 预训练超参数搜索空间
PRETRAIN_SEARCH_SPACE = {
    "method": ["feuler", "eulerian"],  # 序列化方法
    "epochs": [5, 8],
    "batch_size": [32, 64],
    "learning_rate": [1e-4, 2e-4],
    "hidden_size": [512],  # 可以添加 [512, 768]
    # BPE相关参数
    "bpe_enabled": [False, True],
    "bpe_encode_rank_mode": ["all", "topk"],  # 仅在bpe_enabled=True时有效
    "bpe_encode_rank_k": [500, 1000],         # 仅在rank_mode=topk时有效
}

# 微调超参数搜索空间
FINETUNE_SEARCH_SPACE = {
    "method": ["feuler", "eulerian"],
    "task": ["regression"],
    "target_property": ["homo"],
    "finetune_epochs": [15, 20],
    "finetune_batch_size": [16, 32],
    "finetune_learning_rate": [1e-5, 2e-5],
    # BPE相关参数（应与预训练一致）
    "bpe_enabled": [False, True],
    "bpe_encode_rank_mode": ["all", "topk"],
    "bpe_eval_mode": ["all"],  # 评估时使用确定性编码
}

# 搜索策略
SEARCH_STRATEGY = "grid"  # "grid" 或 "random"
MAX_TRIALS = 20  # 仅用于随机搜索

# 结果保存路径
RESULTS_DIR = Path("hyperparam_results")


def create_search_tasks() -> List[Dict[str, Any]]:
    """创建超参数搜索任务列表"""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    tasks = []
    
    if SEARCH_TYPE == "pretrain":
        search_space = PRETRAIN_SEARCH_SPACE
        script_name = "run_pretrain.py"
    else:
        search_space = FINETUNE_SEARCH_SPACE
        script_name = "run_finetune.py"
    
    if SEARCH_STRATEGY == "grid":
        # 网格搜索 - 所有参数组合
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            
            # 生成实验名称
            exp_name_parts = []
            for key, value in params.items():
                if key == "method":
                    exp_name_parts.insert(0, str(value))  # method放在前面
                else:
                    exp_name_parts.append(f"{key}{value}")
            
            task = {
                "script": script_name,
                "params": params,
                "experiment_name": "_".join(exp_name_parts),
                "config_json": None
            }
            tasks.append(task)
    
    elif SEARCH_STRATEGY == "random":
        # 随机搜索 - 随机采样参数组合
        import random
        
        for trial in range(MAX_TRIALS):
            params = {}
            for param_name, param_values in search_space.items():
                params[param_name] = random.choice(param_values)
            
            # 生成实验名称
            exp_name_parts = [f"trial{trial:03d}"]
            for key, value in params.items():
                if key == "method":
                    exp_name_parts.insert(1, str(value))
                elif key not in ["task", "target_property"]:  # 跳过一些常见参数
                    exp_name_parts.append(f"{key}{value}")
            
            task = {
                "script": script_name,
                "params": params,
                "experiment_name": "_".join(exp_name_parts),
                "config_json": None
            }
            tasks.append(task)
    
    print(f"📊 生成了 {len(tasks)} 个超参数搜索任务")
    
    # 保存搜索配置
    config_file = RESULTS_DIR / "search_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "search_type": SEARCH_TYPE,
            "search_strategy": SEARCH_STRATEGY,
            "search_space": search_space,
            "total_tasks": len(tasks),
            "tasks": [{"experiment_name": t["experiment_name"], "params": t["params"]} for t in tasks]
        }, f, indent=2)
    
    print(f"📁 搜索配置已保存到: {config_file}")
    
    return tasks


def run_task(task: Dict[str, Any], gpu_id: int) -> subprocess.Popen:
    """在指定GPU上运行单个任务"""
    cmd = ["python", task["script"]]
    
    # 添加基础参数
    params = task["params"]
    
    # 数据集和实验设置
    cmd.extend([
        "--dataset", DATASET,
        "--experiment_group", EXPERIMENT_GROUP,
        "--experiment_name", task["experiment_name"],
        "--device", f"cuda:{gpu_id}"
    ])
    
    # 添加所有参数
    for key, value in params.items():
        # 特殊处理某些参数
        if key == "bpe_enabled":
            if value:  # 只有为True时才添加该flag
                cmd.append("--bpe_enabled")
        elif key in ["method", "task", "target_property"]:
            cmd.extend([f"--{key}", str(value)])
        elif key.startswith("finetune_") or key.startswith("bpe_"):
            cmd.extend([f"--{key}", str(value)])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"🚀 GPU {gpu_id}: 开始超参数任务 {task['experiment_name']}")
    print(f"   参数: {params}")
    print(f"   命令: {' '.join(cmd)}")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 创建任务专用的日志文件
    log_file = RESULTS_DIR / f"{task['experiment_name']}_gpu{gpu_id}.log"
    
    # 启动进程
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )
    
    return process


def monitor_process(process: subprocess.Popen, task: Dict[str, Any], gpu_id: int) -> Tuple[int, Dict[str, Any]]:
    """监控进程并收集结果"""
    task_name = task["experiment_name"]
    
    # 等待进程完成
    return_code = process.wait()
    
    # 收集结果
    result = {
        "experiment_name": task_name,
        "params": task["params"],
        "return_code": return_code,
        "gpu_id": gpu_id,
        "status": "success" if return_code == 0 else "failed"
    }
    
    # 尝试从日志文件中提取性能指标
    log_file = RESULTS_DIR / f"{task_name}_gpu{gpu_id}.log"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # 简单的性能指标提取（可以根据实际输出格式调整）
                if "最优验证损失" in log_content:
                    # 提取验证损失
                    import re
                    loss_match = re.search(r"最优验证损失: ([\d.]+)", log_content)
                    if loss_match:
                        result["best_val_loss"] = float(loss_match.group(1))
                
                # 可以添加更多指标提取
                
        except Exception as e:
            print(f"⚠️ 无法解析日志文件 {log_file}: {e}")
    
    if return_code == 0:
        print(f"✅ GPU {gpu_id}: 超参数任务 {task_name} 完成")
    else:
        print(f"❌ GPU {gpu_id}: 超参数任务 {task_name} 失败 (退出码: {return_code})")
    
    return return_code, result


def save_results(results: List[Dict[str, Any]]):
    """保存搜索结果"""
    results_file = RESULTS_DIR / "search_results.json"
    
    # 计算统计信息
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]
    
    summary = {
        "total_tasks": len(results),
        "successful_tasks": len(successful_results),
        "failed_tasks": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0
    }
    
    # 如果有性能指标，找出最佳结果
    results_with_metrics = [r for r in successful_results if "best_val_loss" in r]
    if results_with_metrics:
        best_result = min(results_with_metrics, key=lambda x: x["best_val_loss"])
        summary["best_result"] = best_result
    
    full_results = {
        "summary": summary,
        "all_results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"📊 搜索结果已保存到: {results_file}")
    
    # 打印最佳结果
    if "best_result" in summary:
        best = summary["best_result"]
        print(f"\n🏆 最佳结果:")
        print(f"   实验名称: {best['experiment_name']}")
        print(f"   参数: {best['params']}")
        print(f"   验证损失: {best['best_val_loss']:.4f}")


def main():
    """主函数"""
    print("🔍 开始超参数搜索...")
    print(f"搜索类型: {SEARCH_TYPE}")
    print(f"搜索策略: {SEARCH_STRATEGY}")
    print(f"实验组: {EXPERIMENT_GROUP}")
    print(f"数据集: {DATASET}")
    print(f"可用GPU: {GPUS}")
    
    # 创建任务列表
    tasks = create_search_tasks()
    
    if not tasks:
        print("❌ 没有生成任何任务")
        return 1
    
    # 分配任务到GPU
    running_processes = {}  # gpu_id -> (process, task)
    task_queue = tasks.copy()
    all_results = []
    
    def signal_handler(sig, frame):
        print("\n⚠️ 收到中断信号，正在终止所有任务...")
        for gpu_id, (process, task) in running_processes.items():
            if process.poll() is None:  # 进程还在运行
                print(f"🛑 终止 GPU {gpu_id} 上的任务: {task['experiment_name']}")
                process.terminate()
        
        # 保存已完成的结果
        if all_results:
            save_results(all_results)
        
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_time = time.time()
    
    try:
        while task_queue or running_processes:
            # 启动新任务（如果有空闲GPU）
            for gpu_id in GPUS:
                if gpu_id not in running_processes and task_queue:
                    task = task_queue.pop(0)
                    process = run_task(task, gpu_id)
                    running_processes[gpu_id] = (process, task)
            
            # 检查完成的任务
            completed_gpus = []
            for gpu_id, (process, task) in running_processes.items():
                if process.poll() is not None:  # 进程结束
                    return_code, result = monitor_process(process, task, gpu_id)
                    all_results.append(result)
                    completed_gpus.append(gpu_id)
            
            # 移除完成的任务
            for gpu_id in completed_gpus:
                del running_processes[gpu_id]
            
            # 显示进度
            completed = len(all_results)
            total = len(tasks)
            if completed > 0:
                elapsed = time.time() - start_time
                eta = elapsed * (total - completed) / completed
                print(f"📈 进度: {completed}/{total} ({completed/total*100:.1f}%) - 预计剩余: {eta/60:.1f}分钟")
            
            # 等待一下再检查
            if running_processes:
                time.sleep(5)
        
        # 保存最终结果
        save_results(all_results)
        
        # 输出总结
        successful_count = len([r for r in all_results if r["status"] == "success"])
        failed_count = len([r for r in all_results if r["status"] == "failed"])
        
        print("\n" + "="*60)
        print("📊 超参数搜索完成!")
        print("="*60)
        print(f"✅ 成功完成: {successful_count}")
        print(f"❌ 执行失败: {failed_count}")
        print(f"🕒 总耗时: {(time.time() - start_time)/60:.1f} 分钟")
        
        print(f"\n📁 详细结果请查看: {RESULTS_DIR}")
        
        return 0 if failed_count == 0 else 1
        
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        if all_results:
            save_results(all_results)
        return 1


if __name__ == "__main__":
    exit(main())
