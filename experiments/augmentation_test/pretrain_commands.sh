# Experiment P0000: Seq:000 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0000 --mlm_augmentation_methods

# Experiment P0001: Seq:000 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0001 --mlm_augmentation_methods --use_consistency_regularization

# Experiment P0010: Seq:001 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0010 --mlm_augmentation_methods random_truncation

# Experiment P0011: Seq:001 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0011 --mlm_augmentation_methods random_truncation --use_consistency_regularization

# Experiment P0100: Seq:010 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0100 --mlm_augmentation_methods random_swap

# Experiment P0101: Seq:010 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0101 --mlm_augmentation_methods random_swap --use_consistency_regularization

# Experiment P0110: Seq:011 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0110 --mlm_augmentation_methods random_swap random_truncation

# Experiment P0111: Seq:011 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P0111 --mlm_augmentation_methods random_swap random_truncation --use_consistency_regularization

# Experiment P1000: Seq:100 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1000 --mlm_augmentation_methods random_deletion

# Experiment P1001: Seq:100 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1001 --mlm_augmentation_methods random_deletion --use_consistency_regularization

# Experiment P1010: Seq:101 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1010 --mlm_augmentation_methods random_deletion random_truncation

# Experiment P1011: Seq:101 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1011 --mlm_augmentation_methods random_deletion random_truncation --use_consistency_regularization

# Experiment P1100: Seq:110 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1100 --mlm_augmentation_methods random_deletion random_swap

# Experiment P1101: Seq:110 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1101 --mlm_augmentation_methods random_deletion random_swap --use_consistency_regularization

# Experiment P1110: Seq:111 Train:0
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1110 --mlm_augmentation_methods random_deletion random_swap random_truncation

# Experiment P1111: Seq:111 Train:1
python batch_pretrain_simple.py --dataset qm9test --serialization_method graph_seq --bpe_num_merges 2000 --experiment_name P1111 --mlm_augmentation_methods random_deletion random_swap random_truncation --use_consistency_regularization

