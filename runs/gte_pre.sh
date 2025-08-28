python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets zinc --bpe_scenarios all,raw,random \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo \
    --epochs 60 --experiment_group pre_formula \
    --commands_only 
 

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets zinc --bpe_scenarios all,raw,random \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo \
    --epochs 60 --experiment_group pre_formula \
    --commands_stdout 
 