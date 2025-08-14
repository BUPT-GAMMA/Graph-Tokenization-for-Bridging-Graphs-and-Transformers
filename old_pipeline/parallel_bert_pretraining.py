#!/usr/bin/env python3

"""
并行BERT预训练脚本
==================

支持所有序列化方法和BPE压缩选项的并行训练
每个任务占用一张GPU，充分利用多GPU资源

功能：
1. 自动检测可用GPU数量
2. 为每个任务分配独立的GPU
3. 支持所有序列化方法 (feuler, eulerianianian, dfs, bfs, topo, smiles)
4. 支持原始数据和BPE压缩数据
5. 使用项目统一配置
6. 并行训练，提高效率

使用方法：
==========

1. 基本使用（自动检测GPU）：
   python parallel_bert_pretraining.py --dataset qm9

2. 指定GPU数量：
   python parallel_bert_pretraining.py --dataset qm9 --gpu_count 4

3. 指定特定方法：
   python parallel_bert_pretraining.py --dataset qm9 --methods eulerian feuler

4. 只训练原始数据（不训练BPE）：
   python parallel_bert_pretraining.py --dataset qm9 --skip_bpe

5. 只训练BPE数据（不训练原始数据）：
   python parallel_bert_pretraining.py --dataset qm9 --skip_raw

6. 自定义训练参数：
   python parallel_bert_pretraining.py --dataset qm9 --mlm_epochs 10 --mlm_batch_size 16

输出结构（示例，按旧版结构仅供参考）：
==========
models/bert/pretrained/
├── qm9_feuler_raw/
│   └── model.pkl
├── qm9_feuler_bpe/
│   └── model.pkl
├── qm9_euler_raw/
│   └── model.pkl
└── qm9_euler_bpe/
    └── model.pkl

logs/bert/pretrain_bert/
├── qm9_feuler_raw_20250722_223121/
│   ├── logs/
│   └── results/
├── qm9_feuler_bpe_20250722_223122/
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

# 导入BERT预训练模块
from bert_pretrain import pretrain_bert_model

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

def get_available_methods() -> List[str]:
    # """获取所有可用的序列化方法"""
    # try:
    #     methods = list_available_methods(dataset_name)
    #     print(f"📋 可用序列化方法: {methods}")
    #     return methods
    # except Exception as e:
    #     print(f"❌ 获取序列化方法失败: {e}")
    #     # 返回默认方法列表
        return ["feuler", "eulerian", "bfs", "cpp","fcpp","smiles_1","smiles_2","smiles_3","smiles_4"]

def create_training_task_config(
    dataset_name: str,
    method: str,
    use_bpe: bool,
    base_config: ProjectConfig,
    device: str
) -> ProjectConfig:
    """
    为单个训练任务创建配置
    
    Args:
        dataset_name: 数据集名称
        method: 序列化方法
        use_bpe: 是否使用BPE压缩
        base_config: 基础配置
        device: GPU设备
        
    Returns:
        ProjectConfig: 任务专用配置
    """
    # 创建配置副本 - 使用正确的方式
    task_config = ProjectConfig()
    
    # 复制基础配置的所有属性
    for key, value in base_config.__dict__.items():
        setattr(task_config, key, value)
    
    # 设置任务特定参数
    task_config.dataset.name = dataset_name
    task_config.serialization.method = method
    task_config.serialization.bpe.enabled = use_bpe
    task_config.system.device = device
    
    # 设置实验名称（自动生成）- 使用正确的方法
    task_config.experiment_name = None  # 让系统自动生成
    
    return task_config

def run_single_bert_training(args_tuple: Tuple[str, str, bool, ProjectConfig, str, int]) -> Dict[str, Any]:
    """
    运行单个BERT预训练任务
    
    Args:
        args_tuple: (dataset_name, method, use_bpe, base_config, device, task_id)
        
    Returns:
        Dict[str, Any]: 训练结果
    """
    dataset_name, method, use_bpe, base_config, device, task_id = args_tuple
    
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
        task_config = create_training_task_config(
            dataset_name, method, use_bpe, base_config, actual_device
        )
        
        # 设置任务日志
        task_name = f"{method}_{'bpe' if use_bpe else 'raw'}"
        print(f"🔄 [{task_id}] 开始训练: {task_name} (设备: {actual_device})")
        
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
        
        # 运行预训练
        start_time = time.time()
        model_path = pretrain_bert_model(task_config)
        training_time = time.time() - start_time
        
        print(f"✅ [{task_id}] 训练完成: {task_name} (耗时: {training_time:.2f}s)")
        
        return {
            'task_id': task_id,
            'method': method,
            'use_bpe': use_bpe,
            'device': actual_device,
            'success': True,
            'model_path': model_path,
            'training_time': training_time,
            'experiment_name': task_config.get_experiment_name(pipeline='bert')  # 修复：使用正确的方法调用
        }
        
    except Exception as e:
        print(f"❌ [{task_id}] 训练失败: {method}_{'bpe' if use_bpe else 'raw'} - {e}")
        import traceback
        traceback.print_exc()
        return {
            'task_id': task_id,
            'method': method,
            'use_bpe': use_bpe,
            'device': device,
            'success': False,
            'error': str(e)
        }

def run_single_bert_training_with_queue(args_tuple: Tuple[str, str, bool, ProjectConfig, str, int], result_queue: mp.Queue):
    """
    运行单个BERT预训练任务（使用Queue收集结果）
    
    Args:
        args_tuple: (dataset_name, method, use_bpe, base_config, device, task_id)
        result_queue: 结果队列
    """
    result = run_single_bert_training(args_tuple)
    result_queue.put(result)

def run_parallel_bert_training(
    dataset_name: str,
    config: ProjectConfig,
    methods: List[str] = None,
    skip_raw: bool = False,
    skip_bpe: bool = False,
    gpu_count: int = None
) -> List[Dict[str, Any]]:
    """
    并行运行BERT预训练
    
    Args:
        dataset_name: 数据集名称
        config: 基础配置
        methods: 序列化方法列表，None表示使用所有方法
        skip_raw: 是否跳过原始数据训练
        skip_bpe: 是否跳过BPE数据训练
        gpu_count: GPU数量，None表示自动检测
        
    Returns:
        List[Dict[str, Any]]: 训练结果列表
    """
    # 检查GPU可用性
    available_gpu_count, gpu_devices = check_gpu_availability()
    
    if gpu_count is None:
        gpu_count = available_gpu_count
    elif gpu_count > available_gpu_count:
        print(f"⚠️ 请求的GPU数量({gpu_count})超过可用数量({available_gpu_count})，使用可用数量")
        gpu_count = available_gpu_count
    
    # 获取序列化方法
    available_methods = get_available_methods()
    if methods is None:
        methods = available_methods
    else:
        for method in methods:
            if method not in available_methods:
                print(f"⚠️ 指定的序列化方法 {method} 不可用，请检查")
                raise ValueError(f"序列化方法 {method} 不可用")
    
    # 生成训练任务
    tasks = []
    task_id = 0
    
    # 按照raw和bpe来添加
    for method in methods:
        if not skip_bpe:
            device = gpu_devices[task_id % gpu_count] if gpu_count > 0 else "cpu"
            tasks.append((dataset_name, method, True, config, device, task_id))
            task_id += 1
    for method in methods:
        if not skip_raw:
            device = gpu_devices[task_id % gpu_count] if gpu_count > 0 else "cpu"
            tasks.append((dataset_name, method, False, config, device, task_id))
            task_id += 1
    
    if not tasks:
        print("❌ 没有可用的训练任务")
        return []
    
    print(f"🚀 开始并行训练 {len(tasks)} 个任务:")
    for i, (dataset_name, method, use_bpe, config, device, task_id) in enumerate(tasks):
        task_name = f"{method}_{'bpe' if use_bpe else 'raw'}"
        print(f"  任务 {task_id}: {task_name} -> {device}")
    
    # 并行执行 - 使用Process确保真正的并行
    if gpu_count > 0:
        # 使用GPU并行
        processes = []
        result_queue = mp.Queue()
        
        # 创建进程
        for task in tasks:
            p = mp.Process(
                target=run_single_bert_training_with_queue,
                args=(task, result_queue)
            )
            processes.append(p)
        
        # 启动所有进程
        for p in processes:
            p.start()
            # import time
            # time.sleep(2)
        
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

def print_training_summary(results: List[Dict[str, Any]]):
    """打印训练结果摘要"""
    print("\n" + "="*80)
    print("🎉 并行BERT预训练完成！")
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
            print(f"   {task_name}: {result['model_path']} (耗时: {result['training_time']:.2f}s)")
            total_time += result['training_time']
        print(f"   总训练时间: {total_time:.2f}s")
    
    if failed_tasks:
        print(f"\n❌ 失败任务:")
        for result in failed_tasks:
            task_name = f"{result['method']}_{'bpe' if result['use_bpe'] else 'raw'}"
            print(f"   {task_name}: {result['error']}")
    
    print("="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="并行BERT预训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用（自动检测GPU）
  python parallel_bert_pretraining.py --dataset qm9_dgl
  
  # 指定GPU数量
  python parallel_bert_pretraining.py --dataset qm9_dgl --gpu_count 4
  
  # 指定特定方法
  python parallel_bert_pretraining.py --dataset qm9_dgl --methods eulerian feuler
  
  # 只训练原始数据
  python parallel_bert_pretraining.py --dataset qm9_dgl --skip_bpe
  
  # 只训练BPE数据
  python parallel_bert_pretraining.py --dataset qm9_dgl --skip_raw
  
  # 自定义训练参数
  python parallel_bert_pretraining.py --dataset qm9_dgl --mlm_epochs 10 --mlm_batch_size 16
        """
    )
    
    # 基本参数
    parser.add_argument("--dataset", default="qm9", help="数据集名称")
    parser.add_argument("--gpu_count", type=int, help="GPU数量，None表示自动检测")
    parser.add_argument("--methods", nargs="+", help="序列化方法列表")
    parser.add_argument("--skip_raw", action="store_true", help="跳过原始数据训练")
    parser.add_argument("--skip_bpe", action="store_true", help="跳过BPE数据训练")
    
    # BERT训练参数
    parser.add_argument("--mlm_epochs", type=int, help="MLM训练轮数")
    parser.add_argument("--mlm_batch_size", type=int, help="MLM批次大小")
    parser.add_argument("--mlm_learning_rate", type=float, help="MLM学习率")
    parser.add_argument("--d_model", type=int, help="隐藏层大小")
    parser.add_argument("--n_layers", type=int, help="层数")
    parser.add_argument("--n_heads", type=int, help="注意力头数")
    
    args = parser.parse_args()
    
    # 设置日志
    global logger
    logger = setup_logging()
    
    print("🚀 开始并行BERT预训练...")
    
    # 创建基础配置
    config = ProjectConfig()
    
    # 从命令行参数更新配置（统一到新配置层级）
    if args.mlm_epochs:
        config.bert.pretraining.epochs = args.mlm_epochs
    if args.mlm_batch_size:
        config.bert.pretraining.batch_size = args.mlm_batch_size
    if args.mlm_learning_rate:
        config.bert.pretraining.learning_rate = args.mlm_learning_rate
    if args.d_model:
        config.bert.architecture.hidden_size = args.d_model
    if args.n_layers:
        config.bert.architecture.num_hidden_layers = args.n_layers
    if args.n_heads:
        config.bert.architecture.num_attention_heads = args.n_heads
    
    # 验证配置
    config.validate()
    
    # 运行并行训练
    results = run_parallel_bert_training(
        dataset_name=args.dataset,
        config=config,
        methods=args.methods,
        skip_raw=args.skip_raw,
        skip_bpe=args.skip_bpe,
        gpu_count=args.gpu_count
    )
    
    # 打印结果摘要
    print_training_summary(results)
    
    print("🎉 并行BERT预训练完成！")

if __name__ == "__main__":
    main() 