python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix aqsol \
  --datasets aqsol \
  --epochs 100 \
  --batch_size 1024 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --config_json '{"bert": {"architecture": {"max_len_policy": "sigma", "max_len_sigma_k": 3}}}' \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix zinc \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix qm9 \
  --datasets qm9 \
  --epochs 50 \
  --batch_size 1024 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

  
# =====================
# TU datasets - Pretrain
# =====================

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix colors3 \
  --datasets colors3 \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix proteins \
  --datasets proteins \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix synthetic \
  --datasets synthetic \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix mutagenicity \
  --datasets mutagenicity \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix coildel \
  --datasets coildel \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

# 注意：DBLP 词表较大 → 模型更大；减小 batch_size 与训练轮数
python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix dblp \
  --datasets dblp \
  --epochs 20 \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix dd \
  --datasets dd \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix twitter \
  --datasets twitter \
  --epochs 50 \
  --batch_size 128 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \




########finetune#############################



python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix aqsol \
  --datasets aqsol \
  --epochs 60 \
  --batch_size 1024 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --config_json '{"bert": {"architecture": {"max_len_policy": "sigma", "max_len_sigma_k": 3}}}' \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline 

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix zinc \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix qm9 \
  --datasets qm9 \
  --epochs 30 \
  --batch_size 1024 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline
# =====================
# TU datasets - Finetune
# =====================

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix colors3 \
  --datasets colors3 \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix proteins \
  --datasets proteins \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix synthetic \
  --datasets synthetic \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix mutagenicity \
  --datasets mutagenicity \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix coildel \
  --datasets coildel \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

# DBLP：减小 batch_size 与训练轮数
python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix dblp \
  --datasets dblp \
  --epochs 15 \
  --batch_size 64 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix dd \
  --datasets dd \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 818_noaug \
  --exp_prefix twitter \
  --datasets twitter \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation false \
  --log_dir log/subproc/818_noaug \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PnFn" \
  --log_style offline