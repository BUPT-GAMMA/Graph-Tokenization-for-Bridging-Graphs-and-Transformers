python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets molhiv,coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only
    python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets peptides_func --bpe_scenarios all,random,raw --batch_size 32 \
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



#=============================== finetune ====================================================
python batch_finetune_simple.py \
    --encoder bert \
    --datasets molhiv --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --datasets molhiv --bpe_scenarios all,random,raw --learning_rate 5e-6\
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert \
    --datasets peptides_func --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --datasets peptides_func --bpe_scenarios all,random,raw --learning_rate 1e-5\
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
 