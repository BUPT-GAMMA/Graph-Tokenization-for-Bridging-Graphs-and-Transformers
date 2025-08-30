python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets zinc --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --epochs 250 --experiment_group pre_formula_longzinc \
    --commands_stdout
 

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets zinc --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --epochs 250 --experiment_group pre_formula_longzinc \
    --commands_stdout
 