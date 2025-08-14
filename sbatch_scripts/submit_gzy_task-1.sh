#!/bin/bash
#SBATCH --job-name=gzy_task-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/gzy_task-1_%j.out
#SBATCH --error=logs/gzy_task-1_%j.err

mkdir -p logs
echo "INFO: Starting gzy_task-1..."
srun --exclusive -n1 --gres=gpu:1 CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method eulerian --experiment_group test_aug --experiment_name zinc_pre_zinc_eulerian_gaussian_noaug_default --device auto --bpe_encode_rank_mode gaussian --config_json '{"bert": {"pretraining": {"mlm_augmentation_methods": []}}}'
echo "INFO: Finished gzy_task-1."
