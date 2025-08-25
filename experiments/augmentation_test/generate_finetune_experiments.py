#!/usr/bin/env python3
"""
微调增强实验脚本生成器
=====================================

基于选定的预训练实验，生成所有微调增强组合
编码方案：F{pretrain_exp}_{seq4位}{train3位}
- 序列级：deletion(bit0), swap(bit1), truncation(bit2), masking(bit3)  
- 训练级：consistency(bit0), noise(bit1), mixup(bit2)

例如：FP0001_10110101 = 基于P0001预训练的微调实验
"""

import itertools
import argparse
from pathlib import Path
from typing import List


def generate_finetune_experiments():
    """生成128种微调增强组合（独立于预训练实验选择）"""
    
    # 微调可用的序列级增强方法
    seq_augmentations = [
        ('random_deletion', 'deletion'),
        ('random_swap', 'swap'),  
        ('random_truncation', 'truncation'),
        ('sequence_masking', 'masking')
    ]
    
    # 微调可用的训练级增强方法
    train_augmentations = [
        ('use_consistency_regularization', 'consistency'),
        ('use_gaussian_noise', 'noise'),
        ('use_feature_mixup', 'mixup')
    ]
    
    all_experiments = []
    
    print(f"📋 生成全部128种微调实验...")
    
    # 生成所有组合 (2^4 × 2^3 = 128种)
    for seq_combo in itertools.product([0, 1], repeat=len(seq_augmentations)):
        for train_combo in itertools.product([0, 1], repeat=len(train_augmentations)):
            
            # 编码实验名：F{seq4位}{train3位}
            seq_code = ''.join(str(bit) for bit in seq_combo)
            train_code = ''.join(str(bit) for bit in train_combo)
            exp_name = f"F{seq_code}{train_code}"
            
            # 从微调编码推导对应的预训练实验名
            # 预训练编码：P{seq前3位}{train前1位}
            pretrain_seq_code = seq_code[:3]  # 取前3位序列级增强
            pretrain_train_code = train_code[:1]  # 取第1位训练级增强（consistency）
            pretrain_exp = f"P{pretrain_seq_code}{pretrain_train_code}"
            
            # 构建序列级增强配置
            seq_methods = []
            for i, (method, name) in enumerate(seq_augmentations):
                if seq_combo[i] == 1:
                    seq_methods.append(method)
            
            # 构建训练级增强配置  
            train_config = {}
            for i, (config_key, name) in enumerate(train_augmentations):
                train_config[config_key] = bool(train_combo[i])
                
            all_experiments.append({
                'name': exp_name,
                'pretrain_base': pretrain_exp,
                'seq_methods': seq_methods,
                'train_config': train_config,
                'description': f"Pretrain:{pretrain_exp} Seq:{seq_code} Train:{train_code}"
            })
    
    total_exp = len(all_experiments)
    print(f"✅ 总计生成 {total_exp} 个微调实验")
    print(f"   每8个微调实验对应1个预训练实验")
    return all_experiments


def generate_finetune_commands(experiments, output_file: Path):
    """生成微调命令"""
    
    commands = []
    
    for exp in experiments:
        # 构建配置JSON
        config_json = {
            "encoder": {
                "type": "Alibaba-NLP/gte-multilingual-base",
                "reset_weights": True
            },
            "bert": {
                "finetuning": {
                    "regression_augmentation_methods": exp['seq_methods'],
                    "augmentation_config": {}
                }
            }
        }
        
        # 添加训练级增强配置
        for config_key, value in exp['train_config'].items():
            config_json["bert"]["finetuning"]["augmentation_config"][config_key] = value
        
        # 构建命令
        import json
        config_json_str = json.dumps(config_json, separators=(',', ':'))
        
        # 构建基础命令
        cmd_parts = [
            "python run_finetune.py",
            "--dataset zinc",
            "--method feuler",
            f"--experiment_group aug_pretrain",
            f"--experiment_name {exp['name']}",
            "--device auto",
            "--bpe_encode_rank_mode all",
            "--epochs 30",
            "--batch_size 512",
            "--learning_rate 0.0001",
            f"--pretrain_exp_name {exp['pretrain_base']}",
            f"--config_json '{config_json_str}'",
            "--plain_logs"
        ]
                
        commands.append(' '.join(cmd_parts))
    
    # 输出到文件
    with open(output_file, 'w') as f:
        for i, cmd in enumerate(commands):
            f.write(f"CUDA_VISIBLE_DEVICES=0 {cmd}\n")
    
    print(f"✅ 生成了 {len(commands)} 个微调实验命令")
    print(f"📁 输出文件: {output_file}")
    
    return commands


def create_summary(experiments, output_dir: Path):
    """创建实验汇总文件"""
    
    # 按预训练基础分组
    by_pretrain = {}
    for exp in experiments:
        base = exp['pretrain_base']
        if base not in by_pretrain:
            by_pretrain[base] = []
        by_pretrain[base].append(exp)
    
    summary_file = output_dir / "finetune_experiment_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("微调实验汇总\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("编码说明:\n")
        f.write("F{seq4位}{train3位} -> 对应预训练 P{seq前3位}{train前1位}\n")
        f.write("序列级增强: deletion(0), swap(1), truncation(2), masking(3)\n")
        f.write("训练级增强: consistency(0), noise(1), mixup(2)\n\n")
        
        f.write("映射关系说明:\n")
        f.write("每8个微调实验对应1个预训练实验\n")
        f.write("例如: F0000000-F0000111 对应预训练 P0000\n")
        f.write("      F0010000-F0010111 对应预训练 P0010\n\n")
        
        total = 0
        for pretrain_base, exps in sorted(by_pretrain.items()):
            f.write(f"预训练实验 {pretrain_base} 对应的微调实验 ({len(exps)} 个):\n")
            f.write("-" * 40 + "\n")
            
            # 分类统计
            no_aug = [e for e in exps if not e['seq_methods'] and not any(e['train_config'].values())]
            seq_only = [e for e in exps if e['seq_methods'] and not any(e['train_config'].values())]
            train_only = [e for e in exps if not e['seq_methods'] and any(e['train_config'].values())]
            both = [e for e in exps if e['seq_methods'] and any(e['train_config'].values())]
            
            f.write(f"  - 无增强: {len(no_aug)} 个\n")
            f.write(f"  - 仅序列级增强: {len(seq_only)} 个\n") 
            f.write(f"  - 仅训练级增强: {len(train_only)} 个\n")
            f.write(f"  - 混合增强: {len(both)} 个\n")
            f.write("\n")
            total += len(exps)
        
        f.write(f"总实验数: {total}\n")
        f.write(f"预期完成时间: ~{total * 15 / 60:.1f} 小时 (假设每个实验15分钟)\n")
    
    print(f"📄 实验汇总已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="生成全部128种微调实验")
    parser.add_argument("--output_prefix", type=str, default="finetune",
                       help="输出文件前缀")
    
    args = parser.parse_args()
    
    print(f"🚀 生成微调实验脚本...")
    print(f"   模式: 128种微调实验，每8个对应1个预训练实验")
    
    # 生成实验配置
    experiments = generate_finetune_experiments()
    
    # 创建输出目录
    output_dir = Path(__file__).parent
    
    # 生成命令文件
    commands_file = output_dir / f"{args.output_prefix}_commands.sh"
    generate_finetune_commands(experiments, commands_file)
    
    # 创建汇总文件
    create_summary(experiments, output_dir)
    
    print(f"\n🎯 使用方法：")
    print(f"1. 检查命令: head -20 {commands_file}")
    print(f"2. 执行所有实验: bash {commands_file}")
    print(f"3. 收集结果: python collect_finetune_results.py --experiment_group aug_pretrain")
    
    print(f"\n⚠️  注意事项：")
    print(f"   - 确保对应的预训练实验已完成")
    print(f"   - 微调实验使用zinc数据集（回归任务）")
    print(f"   - 预计需要 ~{len(experiments) * 15 / 60:.1f} 小时完成所有实验")


if __name__ == "__main__":
    main()
