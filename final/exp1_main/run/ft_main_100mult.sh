### 对于微调。除了上面的问题可能要注意，还有一点。就是对于molhiv数据集中未经bpe压缩的数据是可以正常训练的。但是对于经过编码压缩的数据，如果不降低learning rate的话会像上面说的一样。在训练一段时间之后突然拔高并且居高不下。或者是很快的过拟合。而且这个部分修改learning rate或者bs等等手段似乎都只能缓解。并不能让他们比raw更高。可能需要考虑怎么解释，或者再修改一下。我有一个猜测，可能是数据量过少。也就是经过编码之后，可能总的token数量会短10倍。导致这个总的token数量过少，那么这个问题也许可以通过启用多重采样来解决。但是需要尝试。暂时就按照mult为1运行，后续再专门测试。

# mutagenicity,molhiv,qm9,twitter
python batch_finetune_simple.py \
    --encoder gte \
    --mult 100 \
    --datasets mutagenicity,qm9 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --learning_rate 1e-5 --mult 100 \
    --datasets molhiv,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only
#normal
# python batch_finetune_simple.py \
#     --encoder gte \
#     --datasets mutagenicity,qm9,twitter --bpe_scenarios all,random,raw \
#     --methods eulerian,feuler,cpp,fcpp \
#     --experiment_group main/1/mult100 --repeat_runs 2\
#     --commands_only

python batch_finetune_simple.py \
    --encoder bert \
    --datasets mutagenicity,molhiv,qm9,twitter --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only


# dblp
python batch_finetune_simple.py \
    --encoder gte \
    --batch_size 32 --learning_rate 1e-5\
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only  
#normal
python batch_finetune_simple.py \
    --encoder bert \
    --batch_size 32 \
    --datasets dblp --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only  

# peptides
# python batch_finetune_simple.py \
#     --encoder gte \
#     --batch_size 32 --learning_rate 1e-5\
#     --datasets peptides_func,peptides_struct --bpe_scenarios all,random,raw \
#     --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
#     --experiment_group main/1/mult100 --repeat_runs 2\
#     --commands_only
python batch_finetune_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets peptides_func,peptides_struct --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only



### 下面是正常运行。
python batch_finetune_simple.py \
    --encoder bert,gte \
    --batch_size 32 \
    --datasets coildel --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only

# zinc,aqsol
python batch_finetune_simple.py \
    --encoder bert,gte \
    --datasets zinc,aqsol,colors3 --bpe_scenarios all,random,raw \
    --methods eulerian,feuler,cpp,fcpp,smiles,topo,dfs,bfs \
    --experiment_group main/1/mult100 --repeat_runs 2\
    --commands_only

