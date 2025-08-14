python batch_pretrain_simple.py \
  --experiment_group test_aug \
  --exp_prefix zinc_pre_ \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --hyperparams_json '[]' \
  --use_augmentation false \
  --log_dir log/subproc/test_aug \
  --commands_only

  python batch_pretrain_simple.py \
  --experiment_group test_aug \
  --exp_prefix test_aug \
  --datasets zinc \
  --gpus 0 \
  --bpe_scenarios raw,all,random,gaussian \
  --hyperparams_json '[]' \
  --use_augmentation true \
  --log_dir log/subproc/test_aug \
  --commands_only