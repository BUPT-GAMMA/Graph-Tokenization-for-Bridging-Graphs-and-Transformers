# Experiment P0000: Seq:000 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0000 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":[],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P0001: Seq:000 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0001 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":[],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P0010: Seq:001 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0010 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_truncation"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P0011: Seq:001 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0011 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_truncation"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P0100: Seq:010 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0100 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_swap"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P0101: Seq:010 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0101 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_swap"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P0110: Seq:011 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0110 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_swap","random_truncation"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P0111: Seq:011 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P0111 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_swap","random_truncation"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P1000: Seq:100 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1000 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P1001: Seq:100 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1001 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P1010: Seq:101 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1010 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_truncation"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P1011: Seq:101 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1011 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_truncation"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P1100: Seq:110 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1100 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_swap"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P1101: Seq:110 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1101 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_swap"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

# Experiment P1110: Seq:111 Train:0
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1110 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_swap","random_truncation"],"augmentation_config":{"use_consistency_regularization":false}}}}' --plain_logs

# Experiment P1111: Seq:111 Train:1
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --dataset zinc --method feuler --experiment_group aug_pretrain --experiment_name P1111 --device auto --bpe_encode_rank_mode all --epochs 20 --batch_size 512 --learning_rate 0.0002 --config_json '{"bert":{"pretraining":{"mlm_augmentation_methods":["random_deletion","random_swap","random_truncation"],"augmentation_config":{"use_consistency_regularization":true}}}}' --plain_logs

