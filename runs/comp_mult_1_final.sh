# 这几个数据集的gte在dfs、bfs、topo上会出现loss飙升后不降。降低学习率有一定的改善，但是仍不能消除问题。
## 经检查，这是bpe编码可能导致的问题：这些数据集的点数较少，所以这三者方法的结果就很短，如果还用较高程度的bpe编码会导致本来就短的序列变得非常短。这样就无法很好的编码。而gte的学习能力可能强于bert，就导致了更显著的过拟合。——后续可能的实验：plot横轴为bpe编码程度的图，比较两者方法的loss或者最终指标。
python batch_pretrain_simple.py \
    --encoder gte \
    --learning_rate 5e-5 \
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,random,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
#normal
python batch_pretrain_simple.py \
    --encoder gte \
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only
python batch_pretrain_simple.py \
    --encoder bert \
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only

## 同上，但是DBLP需要的显存稍微大一点，所以bach size要降低。
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 32 --learning_rate 5e-5 \
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
    --datasets zinc,aqsol,colors3 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 1\
    --commands_only


#=============================== finetune ====================================================

### 对于微调。除了上面的问题可能要注意，还有一点。就是对于molhiv数据集中未经bpe压缩的数据是可以正常训练的。但是对于经过编码压缩的数据，如果不降低learning rate的话会像上面说的一样。在训练一段时间之后突然拔高并且居高不下。或者是很快的过拟合。而且这个部分修改learning rate或者bs等等手段似乎都只能缓解。并不能让他们比raw更高。可能需要考虑怎么解释，或者再修改一下。我有一个猜测，可能是数据量过少。也就是经过编码之后，可能总的token数量会短10倍。导致这个总的token数量过少，那么这个问题也许可以通过启用多重采样来解决。但是需要尝试。暂时就按照mult为1运行，后续再专门测试。

# mutagenicity,molhiv,qm9,twitter
python batch_finetune_simple.py \
    --encoder gte \
    --datasets mutagenicity,qm9,twitter --bpe_scenarios all,random,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --learning_rate 5e-6 \
    --datasets molhiv --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only
#normal
python batch_finetune_simple.py \
    --encoder gte \
    --datasets mutagenicity,qm9,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only

python batch_finetune_simple.py \
    --encoder bert \
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only


# dblp
python batch_finetune_simple.py \
    --encoder gte \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only  
#normal
python batch_finetune_simple.py \
    --encoder gte \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only  
python batch_finetune_simple.py \
    --encoder bert \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only  

# peptides
# python batch_finetune_simple.py \
#     --encoder gte \
#     --batch_size 32 --learning_rate 1e-5\
#     --datasets peptides_func,peptides_struct --bpe_scenarios all,random,raw \
#     --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
#     --experiment_group pre_comp1/mult/1 --repeat_runs 2\
#     --commands_only
python batch_finetune_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets peptides_func,peptides_struct --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only



### 下面是正常运行。
python batch_finetune_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only

# zinc,aqsol
python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets zinc,aqsol,colors3 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group pre_comp1/mult/1 --repeat_runs 2\
    --commands_only

