python batch_pretrain_simple.py \
  --experiment_group test_aug \
  --exp_prefix zinc_pre_ \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --use_augmentation false \
  --log_dir log/subproc/test_aug \
  --commands_only \
  --plain_logs \
  --log_style offline

  python batch_pretrain_simple.py \
  --experiment_group test_aug \
  --exp_prefix zinc_pre_augon_ \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --use_augmentation true \
  --log_dir log/subproc/test_aug \
  --commands_only \
  --plain_logs \
  --log_style offline

  python batch_finetune_simple.py \
  --experiment_group test_aug \
  --exp_prefix zinc_pre_ \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --use_augmentation false \
  --log_dir log/subproc/test_aug \
  --commands_only \
  --plain_logs \
  --log_style offline


  python batch_finetune_simple.py \
  --experiment_group test_aug \
  --exp_prefix zinc_pre_augon_ \
  --save_name_suffix avg \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --use_augmentation true \
  --log_dir log/subproc/test_aug \
  --commands_only \
  --plain_logs \
  --log_style offline