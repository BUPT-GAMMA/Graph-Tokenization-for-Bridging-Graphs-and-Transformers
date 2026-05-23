# Experiment Guide

## Workflow

A typical experiment goes through these steps:

1. **Prepare data** — serialize graphs and train BPE (`prepare_data_new.py`)
2. **Pre-train** — MLM on the token sequences (`run_pretrain.py`)
3. **Fine-tune** — downstream property prediction (`run_finetune.py`)
4. **Aggregate** — collect results across runs (`aggregate_results.py`)

For batch experiments across multiple datasets/methods, use `batch_pretrain_simple.py` and `batch_finetune_simple.py`.

## Running a Single Experiment

```bash
# Step 1: data preparation
python prepare_data_new.py --datasets qm9 --methods feuler --bpe_merges 2000

# Step 2: pre-training
python run_pretrain.py \
    --dataset qm9 --method feuler \
    --experiment_group my_ablation \
    --epochs 100 --batch_size 256

# Step 3: fine-tuning
python run_finetune.py \
    --dataset qm9 --method feuler \
    --experiment_group my_ablation \
    --target_property homo \
    --epochs 200 --batch_size 64
```

The `--experiment_group` flag keeps related runs organized under the same directory in `model/` and `log/`.

## Repeated Runs

By default, each script supports `--repeat_runs N` to run the same experiment N times with different seeds. Aggregated statistics (mean, std, best) are computed automatically and saved as `*_aggregated_stats.json`.

## Batch Experiments

To sweep over multiple datasets and serialization methods:

```bash
python batch_pretrain_simple.py \
    --datasets qm9,zinc,mutagenicity \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1

python batch_finetune_simple.py \
    --datasets qm9,zinc,mutagenicity \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1
```

These scripts distribute jobs across GPUs and run them in parallel.

## Evaluation

Metrics are task-dependent:

- **Regression**: MAE, RMSE, R², Pearson correlation
- **Classification**: accuracy, F1, ROC-AUC
- **Multi-label**: AP (average precision)

All metrics are logged per epoch and saved in `log/{group}/{dataset}-{method}/run_{i}/`.

## Reproducibility

- All random seeds are fixed (default: 42). The config system sets `torch`, `numpy`, and `random` seeds.
- `torch.backends.cudnn.deterministic = True` is set by default.
- A full config snapshot is saved with each run.
- Dataset splits are fixed and loaded from pre-generated indices — no random splitting at training time.

## Fair Comparison Checklist

When comparing methods, make sure:

- Same train/val/test split for all methods
- Same data preprocessing (serialization + BPE settings)
- Same evaluation protocol and metrics
- Multiple runs (at least 3) to report mean ± std
- Same model architecture and training hyperparameters (unless those are the variable being tested)

## Common Pitfalls

- **Data leakage**: using test set statistics during preprocessing
- **Mismatched BPE**: fine-tuning with different BPE settings than pre-training
- **Cherry-picking**: reporting only the best run instead of aggregated statistics
- **Unfixed seeds**: making results non-reproducible
