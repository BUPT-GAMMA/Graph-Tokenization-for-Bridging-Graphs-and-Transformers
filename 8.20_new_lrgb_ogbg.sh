python batch_pretrain_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets molhiv \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets peptides_func \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

python batch_pretrain_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets peptides_struct \
  --epochs 50 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --log_style offline
  # --save_name_prefix "Pn" \

  
########finetune#############################



python batch_finetune_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets molhiv \
  --epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PaFa" \
  --log_style offline 

python batch_finetune_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets peptides_func \
  --epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PaFa" \
  --log_style offline

python batch_finetune_simple.py \
  --experiment_group 820_new_lrgb_ogbg \
  --datasets peptides_struct \
  --epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-5 \
  --gpus 0 \
  --bpe_scenarios all,random,raw \
  --use_augmentation true \
  --log_dir log/subproc/820_new_lrgb_ogbg \
  --commands_only \
  --plain_logs \
  --save_name_prefix "PaFa" \
  --log_style offline