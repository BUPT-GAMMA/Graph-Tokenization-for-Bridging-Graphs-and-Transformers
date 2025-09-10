# 这几个数据集的gte在dfs、bfs、topo上会出现loss飙升后不降。降低学习率有一定的改善，但是仍不能消除问题。
## 经检查，这是bpe编码可能导致的问题：这些数据集的点数较少，所以这三者方法的结果就很短，如果还用较高程度的bpe编码会导致本来就短的序列变得非常短。这样就无法很好的编码。而gte的学习能力可能强于bert，就导致了更显著的过拟合。——后续可能的实验：plot横轴为bpe编码程度的图，比较两者方法的loss或者最终指标。
python batch_pretrain_simple.py \
    --encoder gte \
    --learning_rate 5e-5 --mult 100\
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only
python batch_pretrain_simple.py \
    --encoder gte \
    --learning_rate 5e-5 --mult 100\
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only
#normal
# python batch_pretrain_simple.py \
#     --encoder gte \
#     --mult 100\
#     --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all \
#     --methods eulerian,feuler,cpp,fcpp \
#     --experiment_group main/1/mult100 --repeat_runs 1\
#     --commands_only
python batch_pretrain_simple.py \
    --encoder bert \
    --mult 100\
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only

## 同上，但是DBLP需要的显存稍微大一点，所以bach size要降低。
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 16 --learning_rate 5e-5 --mult 100\
    --datasets dblp --bpe_scenarios all,raw \
    --methods smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only  
python batch_pretrain_simple.py \
    --encoder gte \
    --batch_size 16 --learning_rate 5e-5 --mult 100\
    --datasets dblp --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only  
#normal
# python batch_pretrain_simple.py \
#     --encoder gte \
#     --batch_size 16 --mult 100\
#     --datasets dblp --bpe_scenarios all,raw \
#     --methods eulerian,feuler,cpp,fcpp \
#     --experiment_group main/1/mult100 --repeat_runs 1\
#     --commands_only  
python batch_pretrain_simple.py \
    --encoder bert \
    --batch_size 16 --mult 100\
    --datasets dblp --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only  


### 下面是正常运行。

# peptides_func
python batch_pretrain_simple.py \
    --encoder bert,gte \
    --batch_size 16 --mult 100\
    --datasets peptides_func,coildel --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only


# zinc,aqsol
python batch_pretrain_simple.py \
    --encoder bert,gte --mult 100\
    --datasets aqsol,colors3 --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only
python batch_pretrain_simple.py \
    --encoder bert,gte --mult 100\
    --datasets zinc --bpe_scenarios all,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles \
    --experiment_group main/1/mult100 --repeat_runs 1\
    --commands_only