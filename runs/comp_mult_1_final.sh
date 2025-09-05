# molhiv
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets molhiv --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

# peptides_func
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets peptides_func --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

# qm9,zinc,aqsol
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets qm9,zinc,aqsol --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

# colors3,proteins,mutagenicity,twitter
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets colors3,proteins,mutagenicity,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  

# coildel,dblp
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 50 \
    --datasets coildel,dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  



#=============================== finetune ====================================================
python batch_finetune_simple.py \
    --encoder bert \
    --datasets molhiv --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --datasets molhiv --bpe_scenarios all,random,raw --learning_rate 5e-6\
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert \
    --datasets peptides_func --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --datasets peptides_func --bpe_scenarios all,random,raw --learning_rate 1e-5\
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only


python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets qm9,zinc,aqsol --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets colors3,proteins,mutagenicity,coildel,dblp,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  
 