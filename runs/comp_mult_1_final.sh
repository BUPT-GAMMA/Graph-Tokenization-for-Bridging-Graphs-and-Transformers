# 这几个数据集的gte在dfs、bfs、topo上会出现loss飙升后不降。降低学习率有一定的改善，但是仍不能消除问题。
python batch_pretrain_simple.py \
    --encoder gte \
    --learning_rate 2e-5 \
    --datasets mutagenicity,molhiv,qm9 --bpe_scenarios all,random,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
#normal
python batch_pretrain_simple.py \
    --encoder gte \
    --datasets mutagenicity,molhiv,qm9 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
python batch_pretrain_simple.py \
    --encoder bert \
    --datasets mutagenicity,molhiv,qm9 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

## 同上，但是DBLP需要的显存稍微大一点，所以bach size要降低。
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 32 --learning_rate 2e-5 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  
#normal
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  
python batch_pretrain_simple.py \
    --encoder bert \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  


### 下面是正常运行。

# peptides_func
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets peptides_func,coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only


# zinc,aqsol
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets zinc,aqsol --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

# colors3,twitter
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --datasets colors3,twitter --bpe_scenarios all,random,raw \
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
    --encoder bert,gte \
    --datasets dblp --bpe_scenarios all,random,raw \
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
    --datasets colors3,proteins,mutagenicity,dblp,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only  
 