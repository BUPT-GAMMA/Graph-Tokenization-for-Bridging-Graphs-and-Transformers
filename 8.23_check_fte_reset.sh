# 生成3种预训练命令：BERT + GTE保持权重 + GTE清空权重
python batch_pretrain_simple.py \
    --encoders bert,gte,gte-reset \
    --datasets zinc,molhiv,peptides-func --methods feuler,fcpp,topo --bpe_scenarios random \
    --epochs 30 --experiment_group gte_exploration \
    --plain_logs \
    --log_style offline \
    --use_augmentation true \
    --commands_only

# 生成5种微调命令：BERT + GTE的4种训练方式
python batch_finetune_simple.py \
    --finetune_modes bert,gte-direct,gte-pretrain,gte-reset-direct,gte-reset-pretrain \
    --datasets zinc,molhiv,peptides-func --methods feuler,fcpp,topo --bpe_scenarios random \
    --epochs 30 --experiment_group gte_exploration \
    --plain_logs \
    --log_style offline \
    --use_augmentation true \
    --commands_only
 