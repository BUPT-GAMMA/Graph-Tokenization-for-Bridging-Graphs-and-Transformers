python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets molhiv,coildel,peptides_func --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets qm9,zinc,aqsol --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets colors3,proteins,mutagenicity,coildel,dblp,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only  

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets molhiv,coildel,peptides_func --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets qm9,zinc,aqsol --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets colors3,proteins,mutagenicity,coildel,dblp,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only  
 