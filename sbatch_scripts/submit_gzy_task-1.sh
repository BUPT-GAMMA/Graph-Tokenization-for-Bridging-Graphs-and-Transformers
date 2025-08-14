#!/bin/bash
#SBATCH --job-name=gzy_task-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=a01
#SBATCH --output=logs/gzy_task-1_%j.out
#SBATCH --error=logs/gzy_task-1_%j.err


export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
source /home/fit/shichuan/WORK/miniconda3/envs/pthgnn/bin/activate
echo "INFO: Using Python at $(which python)"

mkdir -p logs
echo "INFO: Starting gzy_task-1..."
srun --exclusive -n1 python run_pretrain.py --dataset zinc --method feuler --experiment_group test_aug --experiment_name zinc_pre_zinc_feuler_raw_noaug_e1 --device auto --bpe_encode_rank_mode none --epochs 1 --batch_size 1024 --learning_rate 0.0005 --config_json '{"bert": {"pretraining": {"mlm_augmentation_methods": []}}, "system": {"log_style": "offline"}}' --plain_logs
echo "INFO: Finished gzy_task-1."
