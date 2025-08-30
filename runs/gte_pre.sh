python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets zinc,molhiv,coildel,peptides_func --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --epochs 50 --experiment_group pre_formula \
    --commands_only
 

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets zinc,molhiv,coildel,peptides_func --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --epochs 50 --experiment_group pre_formula \
    --commands_only 
 