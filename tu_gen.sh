# =====================
# TU datasets - Pretrain
# =====================

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix colors3 \
  --datasets colors3 \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix proteins \
  --datasets proteins \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix synthetic \
  --datasets synthetic \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix mutagenicity \
  --datasets mutagenicity \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix coildel \
  --datasets coildel \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

# 注意：DBLP 词表较大 → 模型更大；减小 batch_size 与训练轮数
python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix dblp \
  --datasets dblp \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix dd \
  --datasets dd \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix twitter \
  --datasets twitter \
  --epochs 50 \
  --batch_size 512 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline


# =====================
# TU datasets - Finetune
# =====================

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix colors3 \
  --datasets colors3 \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix proteins \
  --datasets proteins \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix synthetic \
  --datasets synthetic \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix mutagenicity \
  --datasets mutagenicity \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix coildel \
  --datasets coildel \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

# DBLP：减小 batch_size 与训练轮数
python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix dblp \
  --datasets dblp \
  --epochs 15 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix dd \
  --datasets dd \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix twitter \
  --datasets twitter \
  --epochs 30 \
  --batch_size 512 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline