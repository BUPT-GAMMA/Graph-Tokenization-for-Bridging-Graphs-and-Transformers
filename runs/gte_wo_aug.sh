# 生成3种预训练命令：BERT + GTE保持权重 + GTE清空权重
python batch_pretrain_simple.py \
    --encoders bert,gte,gte-reset \
    --datasets zinc --methods feuler,fcpp,topo --bpe_scenarios all,raw,random \
    --epochs 100 --experiment_group gte_wo_aug \
    --use_augmentation false \
    --commands_stdout\
    --plain_logs  

# 生成5种微调命令：BERT + GTE的4种训练方式
python batch_finetune_simple.py \
    --finetune_modes bert,gte-pretrain,gte-reset-pretrain \
    --datasets zinc --methods feuler,fcpp,topo --bpe_scenarios all,raw,random \
    --epochs 60 --experiment_group gte_wo_aug \
    --use_augmentation false \
    --commands_stdout \
    --plain_logs 

# 生成3种预训练命令：BERT + GTE保持权重 + GTE清空权重
python batch_pretrain_simple.py \
    --encoders bert,gte,gte-reset \
    --datasets zinc --methods feuler,fcpp,topo --bpe_scenarios all,raw,random \
    --epochs 100 --experiment_group gte_wo_aug \
    --use_augmentation true \
    --commands_stdout\
    --plain_logs  

# 生成5种微调命令：BERT + GTE的4种训练方式
python batch_finetune_simple.py \
    --finetune_modes bert,gte-pretrain,gte-reset-pretrain \
    --datasets zinc --methods feuler,fcpp,topo --bpe_scenarios all,raw,random \
    --epochs 60 --experiment_group gte_wo_aug \
    --use_augmentation true \
    --commands_stdout \
    --plain_logs 
