#!/usr/bin/env python3
"""
预训练增强实验脚本生成器
=====================================

生成16种预训练增强组合的实验脚本
编码方案：P{seq3位}{train1位}
- 序列级：deletion(bit0), swap(bit1), truncation(bit2)  
- 训练级：consistency(bit0)

例如：P1011 = deletion+truncation+consistency
"""

import itertools
from pathlib import Path

def generate_pretrain_experiments():
    """生成预训练增强实验的所有组合"""
    
    # 预训练可用的增强方法
    seq_augmentations = [
        ('random_deletion', 'deletion'),
        ('random_swap', 'swap'),  
        ('random_truncation', 'truncation')
    ]
    
    train_augmentations = [
        ('use_consistency_regularization', 'consistency')
    ]
    
    experiments = []
    
    # 生成所有组合 (2^3 × 2^1 = 16种)
    for seq_combo in itertools.product([0, 1], repeat=len(seq_augmentations)):
        for train_combo in itertools.product([0, 1], repeat=len(train_augmentations)):
            
            # 编码实验名
            seq_code = ''.join(str(bit) for bit in seq_combo)
            train_code = ''.join(str(bit) for bit in train_combo)
            exp_name = f"P{seq_code}{train_code}"
            
            # 构建序列级增强配置
            seq_methods = []
            seq_config = {}
            for i, (method, name) in enumerate(seq_augmentations):
                if seq_combo[i] == 1:
                    seq_methods.append(method)
            
            # 构建训练级增强配置  
            train_config = {}
            for i, (config_key, name) in enumerate(train_augmentations):
                train_config[config_key] = bool(train_combo[i])
                
            experiments.append({
                'name': exp_name,
                'seq_methods': seq_methods,
                'train_config': train_config,
                'description': f"Seq:{seq_code} Train:{train_code}"
            })
    
    return experiments

def generate_commands(experiments, output_file):
    """生成预训练命令"""
    
    commands = []
    
    for exp in experiments:
        # 构建配置JSON
        config_json = {
            "encoder": {
                "type": "Alibaba-NLP/gte-multilingual-base",
                "reset_weights": True
            },
            "bert": {
                "pretraining": {
                    "mlm_augmentation_methods": exp['seq_methods'],
                    "augmentation_config": {}
                }
            }
        }
        
        # 添加训练级增强配置
        for config_key, value in exp['train_config'].items():
            config_json["bert"]["pretraining"]["augmentation_config"][config_key] = value
        
        # 构建命令
        import json
        config_json_str = json.dumps(config_json, separators=(',', ':'))
        
        cmd_parts = [
            "python run_pretrain.py",
            "--dataset zinc",
            "--method feuler",
            f"--experiment_group aug_pretrain",
            f"--experiment_name {exp['name']}",
            "--device auto",
            "--bpe_encode_rank_mode all",
            "--epochs 100",  # 适中的epoch数用于快速测试
            "--batch_size 512", 
            "--learning_rate 0.0002",
            f"--config_json '{config_json_str}'",
            "--plain_logs"
        ]
                
        commands.append(' '.join(cmd_parts))
    
    # 输出到文件
    with open(output_file, 'w') as f:
        for i, cmd in enumerate(commands):
            f.write(f"# Experiment {experiments[i]['name']}: {experiments[i]['description']}\n")
            f.write(f"CUDA_VISIBLE_DEVICES=0 {cmd}\n\n")
    
    print(f"✅ 生成了 {len(commands)} 个预训练实验命令")
    print(f"📁 输出文件: {output_file}")
    
    # 打印实验列表
    print("\n📋 实验列表:")
    for exp in experiments:
        seq_desc = ', '.join(exp['seq_methods']) if exp['seq_methods'] else 'None'
        train_desc = ', '.join(k for k, v in exp['train_config'].items() if v) or 'None' 
        print(f"  {exp['name']}: Seq=[{seq_desc}] Train=[{train_desc}]")

if __name__ == "__main__":
    # 生成实验
    experiments = generate_pretrain_experiments()
    
    # 输出命令文件
    output_file = Path(__file__).parent / "pretrain_commands.sh"
    generate_commands(experiments, output_file)
    
    print(f"\n🚀 使用方法：")
    print(f"1. 检查命令: cat {output_file}")
    print(f"2. 执行所有实验: bash {output_file}")
    print(f"3. 收集结果: python collect_pretrain_results.py")
