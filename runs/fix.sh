python batch_finetune_simple.py \
    --encoder gte \
    --datasets molhiv --bpe_scenarios all,random --learning_rate 5e-6\
    --methods eulerian,feuler,cpp,fcpp,smiles \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_only
python batch_finetune_simple.py \
    --encoder gte \
    --datasets peptides_func --bpe_scenarios all,random --learning_rate 1e-5\
    --methods eulerian,feuler,cpp,fcpp,smiles \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_stdout
python batch_finetune_simple.py \
    --encoder gte \
    --datasets peptides_func --bpe_scenarios raw --learning_rate 1e-5 --batch_size 100\
    --methods eulerian,feuler,cpp,fcpp,smiles \
    --experiment_group pre_comp/mult/1 --repeat_runs 1\
    --commands_stdout