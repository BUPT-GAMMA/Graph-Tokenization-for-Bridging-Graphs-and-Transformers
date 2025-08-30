#!/usr/bin/env python3
"""
生成损失函数对比微调指令
==========================

基于预训练模型 molhiv_dfs_all_gte 生成不同损失函数的微调指令：

Focal Loss (多个参数组合测试):
- focal_g2.5_a1.0: γ=2.5, α=1.0 (推荐)
- focal_g3.0_a1.0: γ=3.0, α=1.0 (更激进)
- focal_g2.0_a0.8: γ=2.0, α=0.8 (温和)
- focal_g2.5_a0.5: γ=2.5, α=0.5 (更温和)

加权交叉熵:
- weighted_auto: 自动计算类别权重

标准交叉熵 (基线):
- standard_baseline: 标准交叉熵

实验组: loss_comparison

使用方法:
1. 运行: python test_loss_functions.py
2. 保存输出到文件: python test_loss_functions.py > loss_comparison_commands.sh
3. 在集群上执行: bash loss_comparison_commands.sh
"""

import json
import sys
from pathlib import Path

# 预训练模型配置（基于给定的预训练指令）
PRETRAIN_EXPERIMENT = "molhiv_dfs_all_gte"
DATASET = "molhiv"
METHOD = "dfs"
BPE_MODE = "all"
ENCODER = "gte"

# 微调配置
FINETUNE_EPOCHS = 100
LEARNING_RATE = "5e-5"
DEVICE = "auto"

def generate_finetune_command(loss_method: str, loss_config: dict, param_suffix: str = ""):
    """生成单个微调实验的命令"""

    # 构建实验名称 - 包含参数后缀以区分不同配置
    suffix = f"_{param_suffix}" if param_suffix else ""
    experiment_name = f"{PRETRAIN_EXPERIMENT}_finetune_{loss_method}{suffix}"

    # 将loss_config转换为JSON字符串
    config_json = json.dumps({"task": {"loss_config": loss_config}})

    # 构建命令
    cmd_parts = [
        "python", "run_finetune.py",
        "--dataset", DATASET,
        "--method", METHOD,
        "--experiment_group", "loss_comparison",
        "--experiment_name", experiment_name,
        "--device", DEVICE,
        "--bpe_encode_rank_mode", BPE_MODE,
        "--epochs", str(FINETUNE_EPOCHS),
        "--encoder", ENCODER,
        "--lr", LEARNING_RATE,
        "--pretrain_exp_name", PRETRAIN_EXPERIMENT,
        "--config_json", f"'{config_json}'"
    ]

    return " ".join(cmd_parts), experiment_name

def main():
    """主函数：生成所有损失函数对比实验的命令"""

    print("#!/bin/bash")
    print("#")
    print("# 损失函数对比实验指令集")
    print("# =========================")
    print(f"# 数据集: {DATASET}")
    print(f"# 方法: {METHOD}")
    print(f"# 编码器: {ENCODER}")
    print(f"# BPE模式: {BPE_MODE}")
    print(f"# 预训练模型: {PRETRAIN_EXPERIMENT}")
    print(f"# 微调轮数: {FINETUNE_EPOCHS}")
    print(f"# 学习率: {LEARNING_RATE}")
    print("#")
    print("# 生成时间:", end=" ")
    import datetime
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#")
    print()

    # 定义要测试的损失函数配置
    loss_configs = [
        # Focal Loss - 多个参数组合
        {
            "method": "focal",
            "param_suffix": "g2.5_a1.0",
            "config": {
                'method': 'focal',
                'gamma': 2.5,
                'alpha': 1.0
            },
            "description": "Focal Loss (γ=2.5, α=1.0) - 推荐配置"
        },
        {
            "method": "focal",
            "param_suffix": "g3.0_a1.0",
            "config": {
                'method': 'focal',
                'gamma': 3.0,
                'alpha': 1.0
            },
            "description": "Focal Loss (γ=3.0, α=1.0) - 更激进"
        },
        {
            "method": "focal",
            "param_suffix": "g2.0_a0.8",
            "config": {
                'method': 'focal',
                'gamma': 2.0,
                'alpha': 0.8
            },
            "description": "Focal Loss (γ=2.0, α=0.8) - 温和配置"
        },
        {
            "method": "focal",
            "param_suffix": "g2.5_a0.5",
            "config": {
                'method': 'focal',
                'gamma': 2.5,
                'alpha': 0.5
            },
            "description": "Focal Loss (γ=2.5, α=0.5) - 更温和"
        },
        # 加权交叉熵
        {
            "method": "weighted",
            "param_suffix": "auto",
            "config": {
                'method': 'weighted',
                'auto_weights': True
            },
            "description": "加权交叉熵 (自动计算类别权重)"
        },
        # 标准交叉熵 (基线)
        {
            "method": "standard",
            "param_suffix": "baseline",
            "config": {
                'method': 'standard'
            },
            "description": "标准交叉熵 (基线)"
        }
    ]

    commands = []

    # 生成所有实验命令
    for i, loss_config in enumerate(loss_configs, 1):
        method = loss_config["method"]
        config = loss_config["config"]
        param_suffix = loss_config.get("param_suffix", "")
        description = loss_config.get("description", "")

        # 生成命令
        command, experiment_name = generate_finetune_command(method, config, param_suffix)

        # 输出注释和命令
        print(f"# 实验 {i}: {description}")
        print(f"# 实验名称: {experiment_name}")
        print(command)
        print()

        commands.append(command)

    # 输出统计信息
    print("#" + "=" * 60)
    print(f"# 实验统计: 共生成 {len(commands)} 个实验命令")
    print("# 实验组: loss_comparison")
    print("#")
    print("# 使用方法:")
    print("# 1. 保存到文件: python test_loss_functions.py > loss_comparison_commands.sh")
    print("# 2. 给执行权限: chmod +x loss_comparison_commands.sh")
    print("# 3. 顺序执行: bash loss_comparison_commands.sh")
    print("# 4. 或并行执行: parallel -j 2 < loss_comparison_commands.sh")
    print("#" + "=" * 60)

    return commands

if __name__ == "__main__":
    main()
