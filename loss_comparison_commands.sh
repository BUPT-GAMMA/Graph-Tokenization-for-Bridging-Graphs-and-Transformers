#!/bin/bash
#
# 损失函数对比实验指令集
# =========================
# 数据集: molhiv
# 方法: dfs
# 编码器: gte
# BPE模式: all
# 预训练模型: molhiv_dfs_all_gte
# 微调轮数: 100
# 学习率: 5e-5
#
# 生成时间: 2025-08-30 19:48:26
#

# 实验 1: Focal Loss (γ=2.5, α=1.0) - 推荐配置
# 实验名称: molhiv_dfs_all_gte_finetune_focal_g2.5_a1.0
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_focal_g2.5_a1.0 --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "focal", "gamma": 2.5, "alpha": 1.0}}}'

# 实验 2: Focal Loss (γ=3.0, α=1.0) - 更激进
# 实验名称: molhiv_dfs_all_gte_finetune_focal_g3.0_a1.0
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_focal_g3.0_a1.0 --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "focal", "gamma": 3.0, "alpha": 1.0}}}'

# 实验 3: Focal Loss (γ=2.0, α=0.8) - 温和配置
# 实验名称: molhiv_dfs_all_gte_finetune_focal_g2.0_a0.8
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_focal_g2.0_a0.8 --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "focal", "gamma": 2.0, "alpha": 0.8}}}'

# 实验 4: Focal Loss (γ=2.5, α=0.5) - 更温和
# 实验名称: molhiv_dfs_all_gte_finetune_focal_g2.5_a0.5
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_focal_g2.5_a0.5 --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "focal", "gamma": 2.5, "alpha": 0.5}}}'

# 实验 5: 加权交叉熵 (自动计算类别权重)
# 实验名称: molhiv_dfs_all_gte_finetune_weighted_auto
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_weighted_auto --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "weighted", "auto_weights": true}}}'

# 实验 6: 标准交叉熵 (基线)
# 实验名称: molhiv_dfs_all_gte_finetune_standard_baseline
python run_finetune.py --dataset molhiv --method dfs --experiment_group loss_comparison --experiment_name molhiv_dfs_all_gte_finetune_standard_baseline --device auto --bpe_encode_rank_mode all --epochs 100 --encoder gte --lr 5e-5 --pretrain_exp_name molhiv_dfs_all_gte --config_json '{"task": {"loss_config": {"method": "standard"}}}'

#============================================================
# 实验统计: 共生成 6 个实验命令
# 实验组: loss_comparison
#
# 使用方法:
# 1. 保存到文件: python test_loss_functions.py > loss_comparison_commands.sh
# 2. 给执行权限: chmod +x loss_comparison_commands.sh
# 3. 顺序执行: bash loss_comparison_commands.sh
# 4. 或并行执行: parallel -j 2 < loss_comparison_commands.sh
#============================================================
