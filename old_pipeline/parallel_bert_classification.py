#!/usr/bin/env python3
"""
并行BERT分类微调脚本
==================

支持所有序列化方法和BPE压缩选项的并行分类微调
每个任务占用一张GPU，充分利用多GPU资源
自动加载对应的预训练模型

功能：
1. 自动检测可用GPU数量
2. 为每个任务分配独立的GPU
3. 支持所有序列化方法 (feuler, eulerian, dfs, bfs, topo, smiles)
4. 支持原始数据和BPE压缩数据
5. 自动加载对应的预训练模型
6. 使用项目统一配置
7. 并行微调，提高效率
8. 支持多分类任务

使用方法：
==========

1. 基本使用（自动检测GPU）：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3

2. 指定GPU数量：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3 --gpu_count 4

3. 指定特定方法：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3 --methods eulerian feuler

4. 只微调原始数据（不微调BPE）：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3 --skip_bpe

5. 只微调BPE数据（不微调原始数据）：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3 --skip_raw

6. 自定义微调参数：
   python parallel_bert_classification.py --dataset qm9 --num_classes 3 --finetune_epochs 20 --finetune_batch_size 16

输出结构：
==========
model/finetune_bert/
├── qm9/
│   └── {experiment_name}/
│       ├── feuler_raw/
│       │   └── model.pkl
│       ├── feuler_bpe/
│       │   └── model.pkl
│       ├── eulerian_raw/
│       │   └── model.pkl
│       └── eulerian_bpe/
│           └── model.pkl

log/finetune_bert/
├── {experiment_name}/
│   ├── logs/
│   └── results/
└── ...
"""


# 在所有其他导入之前执行！
import sys
import threading

# 强制在主线程提前导入 TensorBoard
sys.modules["_tb_import_lock"] = threading.Lock()  # 防止多线程导入竞争

# 显式预加载关键模块
import tensorboard
from torch.utils.tensorboard import SummaryWriter

import os
import random
import sys
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

# 设置多进程启动方法为spawn（CUDA兼容）
mp.set_start_method('spawn', force=True)

import torch
import numpy as np

# 导入项目模块
from config import ProjectConfig

# 导入BERT分类微调模块
from bert_classification import finetune_bert_classification

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_gpu_availability() -> Tuple[int, List[str]]:
    """
    检查可用GPU数量
    
    Returns:
        Tuple[int, List[str]]: (GPU数量, GPU设备列表)
    """
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，将使用CPU训练")
        return 0, []
    
    gpu_count = torch.cuda.device_count()
    gpu_devices = [f"cuda:{i}" for i in range(gpu_count)]
    
    print(f"🔍 检测到 {gpu_count} 个GPU设备:")
    for i, device in enumerate(gpu_devices):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"   GPU {i}: {gpu_name}")
    
    return gpu_count, gpu_devices

def get_available_methods(dataset_name: str = "qm9") -> List[str]:
    """获取所有可用的序列化方法"""
    # try:
    #     methods = list_available_methods(dataset_name)
    #     print(f"📋 可用序列化方法: {methods}")
    #     return methods
    # except Exception as e:
    #     print(f"❌ 获取序列化方法失败: {e}")
    #     # 返回默认方法列表
    return ["feuler", "eulerian", "bfs", "cpp","fcpp",]

def get_pretrained_model_path(
    dataset_name: str,
    method: str,
    use_bpe: bool,
    base_config: ProjectConfig
) -> Path:
    """
    查找对应的预训练模型路径
    
    新的路径结构: model/pretrain_bert/{dataset}/{experiment_name}/{method}_{raw/bpe}/model.pkl
    由于微调和预训练的时间不同，实验名称会不同，需要搜索最新的预训练模型
    
    Args:
        dataset_name: 数据集名称
        method: 序列化方法
        use_bpe: 是否使用BPE压缩
        base_config: 基础配置
        
    Returns:
        Path: 预训练模型路径
    """
    # 构造方法目录名
    bpe_suffix = "bpe" if use_bpe else "raw"
    method_dir = f"{method}_{bpe_suffix}"
    
    # 获取预训练模型根目录：model/pretrain_bert/{dataset_name}/
    models_root = Path("model/pretrain_bert") / dataset_name
    
    if not models_root.exists():
        raise FileNotFoundError(f"预训练模型根目录不存在: {models_root}")
    
    # 搜索匹配的实验目录（包含时间戳）
    matching_paths = []
    for experiment_dir in models_root.iterdir():
        if experiment_dir.is_dir():
            # 实际路径结构: {timestamp}/{dataset}_{method}_{bpe/raw}/{method}_{bpe/raw}/model.pkl
            # 构造完整的方法目录路径
            full_method_dir = f"{dataset_name}_{method}_{bpe_suffix}"
            method_path = experiment_dir / full_method_dir / method_dir
            if method_path.exists() and method_path.is_dir():
                # 检查是否有model.pkl文件
                model_file = method_path / "model.pkl"
                if model_file.exists():
                    matching_paths.append(model_file)
    
    if not matching_paths:
        raise FileNotFoundError(f"未找到匹配的预训练模型: {method_dir} 在 {models_root}")
    
    if len(matching_paths) == 1:
        # 找到唯一匹配的模型
        return matching_paths[0]
    else:
        # 找到多个匹配的模型，选择最新的（按实验目录名排序，时间戳在前面）
        # 实验名称格式: MMdd_HHMM/dataset_method_bpe，时间戳在开头
        matching_paths.sort(key=lambda x: x.parent.parent.parent.name, reverse=True)
        experiment_name = matching_paths[0].parent.parent.parent.name
        print(f"⚠️ 找到多个匹配的预训练模型，选择最新的实验: {experiment_name}")
        return matching_paths[0]

def create_finetuning_task_config(
    dataset_name: str,
    method: str,
    use_bpe: bool,
    base_config: ProjectConfig,
    device: str
) -> ProjectConfig:
    """
    为单个微调任务创建配置
    
    Args:
        dataset_name: 数据集名称
        method: 序列化方法
        use_bpe: 是否使用BPE压缩
        base_config: 基础配置
        device: GPU设备
        
    Returns:
        ProjectConfig: 任务专用配置
    """
    # 创建配置副本
    task_config = ProjectConfig()
    
    # 复制基础配置的所有属性
    for key, value in base_config.__dict__.items():
        setattr(task_config, key, value)
    
    # 设置任务特定参数
    task_config.dataset.name = dataset_name
    task_config.serialization.method = method
    task_config.serialization.bpe.enabled = use_bpe
    task_config.system.device = device
    
    # 设置实验名称（自动生成）
    task_config.experiment_name = None  # 让系统自动生成
    
    return task_config

def run_single_bert_classification_finetuning(args_tuple: Tuple[str, str, bool, ProjectConfig, str, int, int]) -> Dict[str, Any]:
    """
    运行单个BERT分类微调任务
    
    Args:
        args_tuple: (dataset_name, method, use_bpe, base_config, device, task_id, num_classes)
        
    Returns:
        Dict[str, Any]: 微调结果
    """
    dataset_name, method, use_bpe, base_config, device, task_id, num_classes = args_tuple
    
    try:
        # 设置CUDA设备（必须在导入PyTorch之前设置）
        if device and device.startswith("cuda"):
            device_id = device.split(":")[-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            # 在多进程环境中，重新设置设备为cuda:0（因为CUDA_VISIBLE_DEVICES已经限制了可见设备）
            actual_device = "cuda:0"
        else:
            actual_device = device
        
        # 创建任务配置
        task_config = create_finetuning_task_config(
            dataset_name, method, use_bpe, base_config, actual_device
        )
        
        # 获取对应的预训练模型路径
        pretrained_path = get_pretrained_model_path(dataset_name, method, use_bpe, base_config)
        
        # 检查预训练模型是否存在
        if not pretrained_path.exists():
            error_msg = f"预训练模型不存在: {pretrained_path}"
            print(f"❌ [{task_id}] {error_msg}")
            return {
                'task_id': task_id,
                'method': method,
                'use_bpe': use_bpe,
                'device': device,
                'num_classes': num_classes,
                'success': False,
                'error': error_msg
            }
        
        # 设置任务日志
        task_name = f"{method}_{'bpe' if use_bpe else 'raw'}"
        print(f"🔄 [{task_id}] 开始分类微调: {task_name} (类别数: {num_classes}, 设备: {actual_device})")
        print(f"📂 [{task_id}] 加载预训练模型: {pretrained_path}")
        
        # 验证GPU设置是否正确
        if actual_device.startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                # 获取实际的物理GPU ID（通过CUDA_VISIBLE_DEVICES映射）
                physical_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                print(f"🔍 [{task_id}] 物理GPU: {physical_gpu_id}, 逻辑GPU: {current_device} - {device_name}")
            else:
                print(f"⚠️ [{task_id}] CUDA不可用")
        
        # 运行分类微调
        start_time = time.time()
        model_path = finetune_bert_classification(task_config, num_classes, str(pretrained_path))
        training_time = time.time() - start_time
        
        print(f"✅ [{task_id}] 分类微调完成: {task_name} (类别数: {num_classes}, 耗时: {training_time:.2f}s)")
        
        return {
            'task_id': task_id,
            'method': method,
            'use_bpe': use_bpe,
            'device': actual_device,
            'num_classes': num_classes,
            'success': True,
            'model_path': model_path,
            'pretrained_path': str(pretrained_path),
            'training_time': training_time,
            'experiment_name': task_config.get_experiment_name(pipeline='bert')
        }
        
    except Exception as e:
        print(f"❌ [{task_id}] 分类微调失败: {method}_{'bpe' if use_bpe else 'raw'} (类别数: {num_classes}) - {e}")
        import traceback
        traceback.print_exc()
        return {
            'task_id': task_id,
            'method': method,
            'use_bpe': use_bpe,
            'device': device,
            'num_classes': num_classes,
            'success': False,
            'error': str(e)
        }

def run_single_bert_classification_finetuning_with_queue(args_tuple: Tuple[str, str, bool, ProjectConfig, str, int, int], result_queue: mp.Queue):
    """
    运行单个BERT分类微调任务（使用Queue收集结果）
    
    Args:
        args_tuple: (dataset_name, method, use_bpe, base_config, device, task_id, num_classes)
        result_queue: 结果队列
    """
    result = run_single_bert_classification_finetuning(args_tuple)
    result_queue.put(result)

def run_parallel_bert_classification_finetuning(
    dataset_name: str,
    config: ProjectConfig,
    num_classes: int,
    methods: List[str] = None,
    skip_raw: bool = False,
    skip_bpe: bool = False,
    gpu_count: int = None
) -> List[Dict[str, Any]]:
    """
    并行运行BERT分类微调
    
    Args:
        dataset_name: 数据集名称
        config: 基础配置
        num_classes: 类别数量
        methods: 序列化方法列表，None表示使用所有方法
        skip_raw: 是否跳过原始数据微调
        skip_bpe: 是否跳过BPE数据微调
        gpu_count: GPU数量，None表示自动检测
        
    Returns:
        List[Dict[str, Any]]: 微调结果列表
    """
    # 检查GPU可用性
    available_gpu_count, gpu_devices = check_gpu_availability()
    
    if gpu_count is None:
        gpu_count = available_gpu_count
    elif gpu_count > available_gpu_count:
        print(f"⚠️ 请求的GPU数量({gpu_count})超过可用数量({available_gpu_count})，使用可用数量")
        gpu_count = available_gpu_count
    
    # 获取序列化方法
    available_methods = get_available_methods(dataset_name)
    if methods is None:
        methods = available_methods
    else:
        for method in methods:
            if method not in available_methods:
                print(f"⚠️ 指定的序列化方法 {method} 不可用，请检查")
                raise ValueError(f"序列化方法 {method} 不可用")
    
    # 生成微调任务
    tasks = []
    task_id = 0
    
    # 按照raw和bpe来添加
    for method in methods:
        if not skip_bpe:
            device = gpu_devices[task_id % gpu_count] if gpu_count > 0 else "cpu"
            tasks.append((dataset_name, method, True, config, device, task_id, num_classes))
            task_id += 1
    for method in methods:    
        if not skip_raw:
            device = gpu_devices[task_id % gpu_count] if gpu_count > 0 else "cpu"
            tasks.append((dataset_name, method, False, config, device, task_id, num_classes))
            task_id += 1
    
    if not tasks:
        print("❌ 没有可用的微调任务")
        return []
    
    print(f"🚀 开始并行分类微调 {len(tasks)} 个任务 (类别数: {num_classes}):")
    for i, (dataset_name, method, use_bpe, config, device, task_id, num_classes) in enumerate(tasks):
        task_name = f"{method}_{'bpe' if use_bpe else 'raw'}"
        pretrained_path = get_pretrained_model_path(dataset_name, method, use_bpe, config)
        print(f"  任务 {task_id}: {task_name} (类别数: {num_classes}) -> {device}")
        print(f"    预训练模型: {pretrained_path}")
        if not pretrained_path.exists():
            print(f"    ⚠️ 预训练模型不存在，该任务将失败")
    
    # 并行执行 - 使用Process确保真正的并行
    if gpu_count > 0:
        # 使用GPU并行
        processes = []
        result_queue = mp.Queue()
        
        # 创建进程
        for task in tasks:
            p = mp.Process(
                target=run_single_bert_classification_finetuning_with_queue,
                args=(task, result_queue)
            )
            processes.append(p)
        
        # 启动所有进程
        for p in processes:
            p.start()
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 按task_id排序结果
        results.sort(key=lambda x: x['task_id'])
        
    else:
        raise ValueError("GPU数量必须大于0")
    
    return results

def print_finetuning_summary(results: List[Dict[str, Any]]):
    """打印微调结果摘要"""
    print("\n" + "="*80)
    print("🎉 并行BERT分类微调完成！")
    print("="*80)
    
    successful_tasks = [r for r in results if r['success']]
    failed_tasks = [r for r in results if not r['success']]
    
    print(f"📊 总体统计:")
    print(f"   总任务数: {len(results)}")
    print(f"   成功任务: {len(successful_tasks)}")
    print(f"   失败任务: {len(failed_tasks)}")
    
    if successful_tasks:
        print(f"\n✅ 成功任务:")
        total_time = 0
        for result in successful_tasks:
            task_name = f"{result['method']}_{'bpe' if result['use_bpe'] else 'raw'}"
            print(f"   {task_name} (类别数: {result['num_classes']}): {result['model_path']} (耗时: {result['training_time']:.2f}s)")
            print(f"     预训练模型: {result['pretrained_path']}")
            total_time += result['training_time']
        print(f"   总微调时间: {total_time:.2f}s")
    
    if failed_tasks:
        print(f"\n❌ 失败任务:")
        for result in failed_tasks:
            task_name = f"{result['method']}_{'bpe' if result['use_bpe'] else 'raw'}"
            print(f"   {task_name} (类别数: {result['num_classes']}): {result['error']}")
    
    print("="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="并行BERT分类微调脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（自动检测GPU）
  python parallel_bert_classification.py --dataset qm9 --num_classes 3
  
  # 指定GPU数量
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --gpu_count 4
  
  # 指定特定方法
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --methods eulerian feuler
  
  # 只微调原始数据
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --skip_bpe
  
  # 只微调BPE数据
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --skip_raw
  
  # 指定目标属性
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --target_property homo
  
  # 自定义微调参数
  python parallel_bert_classification.py --dataset qm9 --num_classes 3 --finetune_epochs 20 --finetune_batch_size 16
        """
    )
    
    # 基本参数
    parser.add_argument("--dataset", default="qm9", help="数据集名称")
    parser.add_argument("--num_classes", type=int, required=True, help="类别数量")
    parser.add_argument("--gpu_count", type=int, help="GPU数量，None表示自动检测")
    parser.add_argument("--methods", nargs="+", help="序列化方法列表")
    parser.add_argument("--skip_raw", action="store_true", help="跳过原始数据微调")
    parser.add_argument("--skip_bpe", action="store_true", help="跳过BPE数据微调")
    
    # BERT微调参数
    parser.add_argument("--finetune_epochs", type=int, help="微调训练轮数")
    parser.add_argument("--finetune_batch_size", type=int, help="微调批次大小")
    parser.add_argument("--finetune_learning_rate", type=float, help="微调学习率")
    parser.add_argument("--early_stopping_patience", type=int, help="早停耐心值")
    parser.add_argument("--pooling_method", type=str, choices=['cls', 'mean', 'max'], help="池化方法")
    parser.add_argument("--target_property", type=str, default="homo", help="目标属性 (默认: homo)")
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging()
    
    print("🚀 开始并行BERT分类微调...")
    print(f"🎯 分类任务配置: 类别数={args.num_classes}, 目标属性={args.target_property}")
    
    # 创建基础配置
    config = ProjectConfig()
    
    # 从命令行参数更新配置
    if args.finetune_epochs:
        config.bert.finetuning.epochs = args.finetune_epochs
    if args.finetune_batch_size:
        config.bert.finetuning.batch_size = args.finetune_batch_size
    if args.finetune_learning_rate:
        config.bert.finetuning.learning_rate = args.finetune_learning_rate
    if args.early_stopping_patience:
        config.bert.finetuning.early_stopping_patience = args.early_stopping_patience
    if args.pooling_method:
        config.bert.architecture.pooling_method = args.pooling_method
    if args.target_property:
        config.task.target_property = args.target_property
        print(f"🎯 设置目标属性: {args.target_property}")
    else:
        print(f"🎯 使用默认目标属性: {config.task.target_property}")
    
    # 验证配置
    config.validate()
    
    # 运行并行微调
    results = run_parallel_bert_classification_finetuning(
        dataset_name=args.dataset,
        config=config,
        num_classes=args.num_classes,
        methods=args.methods,
        skip_raw=args.skip_raw,
        skip_bpe=args.skip_bpe,
        gpu_count=args.gpu_count
    )
    
    # 打印结果摘要
    print_finetuning_summary(results)
    
    print("🎉 并行BERT分类微调完成！")

if __name__ == "__main__":
    main()
