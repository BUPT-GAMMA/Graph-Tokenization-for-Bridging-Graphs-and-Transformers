python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix aqsol \
  --datasets aqsol \
  --epochs 100 \
  --batch_size 1024 \
  --learning_rate 2e-4 \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --config_json '{"bert": {"architecture": {"max_len_policy": "sigma", "max_len_sigma_k": 3}}}' \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix zinc \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_pretrain_simple.py \
  --experiment_group 815_base \
  --exp_prefix qm9 \
  --datasets qm9 \
  --epochs 50 \
  --batch_size 1024 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline




python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix aqsol \
  --datasets aqsol \
  --epochs 60 \
  --batch_size 1024 \
  --learning_rate 2e-5 \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --config_json '{"bert": {"architecture": {"max_len_policy": "sigma", "max_len_sigma_k": 3}}}' \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline 

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix zinc \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 815_base \
  --exp_prefix qm9 \
  --datasets qm9 \
  --epochs 30 \
  --batch_size 1024 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all \
  --use_augmentation true \
  --log_dir log/subproc/815_base \
  --commands_only \
  --plain_logs \
  --log_style offline