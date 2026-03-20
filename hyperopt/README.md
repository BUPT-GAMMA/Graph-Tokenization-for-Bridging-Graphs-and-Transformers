# Hyperparameter Search

This directory contains the maintained Optuna workflow for TokenizerGraph. This is the primary hyperparameter-search documentation entrypoint.

## What Framework We Use

The search stack is built on `Optuna`:

- sampler: `TPESampler`
- pruner: `MedianPruner` or `PercentilePruner`
- shared study storage: `JournalStorage` with `JournalFileBackend`

This design keeps the search state in a local file, supports resume, and can be reused by multiple workers as long as they point to the same journal file.

Official references:

- Optuna docs: <https://optuna.readthedocs.io/en/stable/>
- Journal storage reference: <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html>

## Directory Layout

```text
hyperopt/
├── README.md
├── journal/                  # Optuna journal files, not committed
├── results/                  # exported candidate JSON files, not committed
└── scripts/
    ├── common.py
    ├── large_batch_search.py
    ├── extract_best_params_for_finetuning.py
    ├── finetune_with_pretrain_options.py
    └── analyze_optuna_results.py
```

## Maintained Entry Points

- `hyperopt/scripts/large_batch_search.py`
  - pretraining hyperparameter search
  - optional finetuning search directly from top-K pretraining trials
- `hyperopt/scripts/extract_best_params_for_finetuning.py`
  - exports reusable pretraining candidates from an Optuna journal
- `hyperopt/scripts/finetune_with_pretrain_options.py`
  - finetuning search over exported pretraining options
- `hyperopt/scripts/analyze_optuna_results.py`
  - compact study summary and best-trial inspection

## What Can Be Searched

The maintained scripts expose several search dimensions directly from the CLI, and they are not limited to learning rate.

Directly configurable search-space arguments already exposed by the scripts include:

- `--methods`
- `--bpe_mode`
- `--bpe_backend`
- `--encoder`
- `--batch_sizes`
- `--finetune_batch_sizes`
- `--lr_min`, `--lr_max`
- `--finetune_lr_min`, `--finetune_lr_max`
- `--wd_min`, `--wd_max`
- `--grad_norm_min`, `--grad_norm_max`
- `--mask_prob_min`, `--mask_prob_max`
- `--warmup_min`, `--warmup_max`
- `--pretrain_epochs`, `--finetune_epochs`
- `--target_property`

In addition, base config values can be changed through `--config_json`, for example:

- model size
- attention heads / hidden size / layer count
- early stopping patience
- `save_models`
- data-loader worker settings
- multiple-sampling settings
- augmentation settings

If you need to search a parameter that is not currently exposed as a `trial.suggest_*` field, add it in the corresponding script. The maintained scripts are intentionally small so search spaces can be extended without touching the main training entrypoints.

## Backend Notes

- `--bpe_backend cpp` is the default for normal runs and requires:

```bash
python setup.py build_ext --inplace
```

- `--bpe_backend python` is available as a fallback when `_cpp_bpe` is unavailable.
- The current Python fallback is intended for deterministic smoke/debug runs and currently supports `--bpe_mode all` or `--bpe_mode none`.

## Prerequisites

Before running search, make sure:

1. the dataset already exists under `data/<dataset>/`, either from the released bundle or from the conversion scripts
2. `prepare_data_new.py` has already created the serialized cache and vocab for the dataset/method you want to search
3. CUDA is available; the maintained search scripts fail fast if CUDA is unavailable

Current dataset setup references:

- `README.md`
- `README_zh.md`
- `scripts/dataset_conversion/README.md`

## Recommended Workflow

### 1. Search pretraining hyperparameters

Example:

```bash
python hyperopt/scripts/large_batch_search.py \
  --dataset zinc \
  --methods fcpp \
  --bpe_mode all \
  --bpe_backend cpp \
  --encoder gte \
  --stage pretrain \
  --batch_sizes 128,256,512 \
  --pretrain_epochs 50 \
  --pretrain_trials 20 \
  --journal_file hyperopt/journal/large_batch.db
```

The study name is stable:

- pretrain study example: `hyperopt_pretrain_zinc_all`
- pretrain checkpoint example: `search_zinc_all_pt_003`

Artifacts:

- journal file, for example `hyperopt/journal/large_batch.db`
- pretrain checkpoints under `model/<experiment_group>/<experiment_name>/run_0/best/`

### 2. Export reusable pretrain candidates

```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/large_batch.db \
  --dataset zinc \
  --bpe_mode all \
  --output_dir hyperopt/results
```

This writes:

- `hyperopt/results/best_pretrain_params_for_finetuning.json`

The exported JSON contains:

- overall top-K pretraining trials
- per-method top trials
- trial number, method, checkpoint path, experiment name, raw Optuna params, and timing metadata

### 3. Search finetuning hyperparameters

```bash
python hyperopt/scripts/finetune_with_pretrain_options.py \
  --dataset zinc \
  --bpe_mode all \
  --bpe_backend cpp \
  --encoder gte \
  --options_file hyperopt/results/best_pretrain_params_for_finetuning.json \
  --journal_file hyperopt/journal/finetune_with_options.db \
  --trials 20 \
  --epochs 50
```

For regression datasets that need an explicit target:

```bash
python hyperopt/scripts/finetune_with_pretrain_options.py \
  --dataset qm9test \
  --bpe_mode all \
  --bpe_backend python \
  --encoder bert \
  --target_property homo \
  --options_file hyperopt/results/best_pretrain_params_for_finetuning.json \
  --trials 5 \
  --epochs 1
```

The finetuning search normalizes every target to Optuna `minimize`:

- minimizing metrics like `mae` and `rmse` are used directly
- maximizing metrics like `accuracy`, `roc_auc`, and `ap` are negated before being returned to Optuna

The raw metric value is still stored in `user_attrs`, so study summaries remain readable.

### 4. Inspect studies

```bash
python hyperopt/scripts/analyze_optuna_results.py \
  --journal hyperopt/journal/large_batch.db \
  --dataset zinc \
  --bpe_mode all \
  --stage pretrain
```

## Verified Smoke Route

The following route was verified on the current repository state on 2026-03-21.

### Pretrain smoke

```bash
python hyperopt/scripts/large_batch_search.py \
  --dataset qm9test \
  --methods feuler \
  --bpe_mode all \
  --bpe_backend python \
  --encoder bert \
  --target_property homo \
  --stage pretrain \
  --batch_sizes 8 \
  --pretrain_epochs 1 \
  --pretrain_trials 1 \
  --journal_file hyperopt/journal/qm9test_smoke_python2.db \
  --config_json '{"encoder":{"reset_weights":false},"bert":{"architecture":{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":256,"max_seq_length":512,"max_position_embeddings":512},"pretraining":{"early_stopping_patience":1},"finetuning":{"early_stopping_patience":1,"save_models":true}}}'
```

Observed result:

- pretrain study completed
- best pretrain objective: `5.727606278987019`
- checkpoint written to `model/hyperopt_pretrain_qm9test_all/search_qm9test_all_pt_000/run_0/best/`

### Export candidates

```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/qm9test_smoke_python2.db \
  --dataset qm9test \
  --bpe_mode all \
  --output_dir hyperopt/results/qm9test_smoke
```

Observed result:

- exported file: `hyperopt/results/qm9test_smoke/best_pretrain_params_for_finetuning.json`

### Finetune smoke

```bash
python hyperopt/scripts/finetune_with_pretrain_options.py \
  --dataset qm9test \
  --bpe_mode all \
  --bpe_backend python \
  --encoder bert \
  --target_property homo \
  --options_file hyperopt/results/qm9test_smoke/best_pretrain_params_for_finetuning.json \
  --journal_file hyperopt/journal/qm9test_finetune_smoke_python2.db \
  --batch_sizes 8 \
  --epochs 1 \
  --trials 1 \
  --config_json '{"encoder":{"reset_weights":false},"bert":{"architecture":{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"intermediate_size":256,"max_seq_length":512,"max_position_embeddings":512},"pretraining":{"early_stopping_patience":1},"finetuning":{"early_stopping_patience":1,"save_models":true}}}'
```

Observed result:

- finetune study completed
- best finetune objective: `0.27749751177908605`

Additional verification:

- fast follow-up finetune verification with `--batch_sizes 256` completed successfully
- `src/utils/info_display.py` now reports the correct pretrain/finetune config branch
- `src/training/finetune_pipeline.py` now writes `finetune_metrics.json` cleanly under Optuna-driven runs
- the maintained scripts can run end to end without `_cpp_bpe` when `--bpe_backend python` is selected

## Related Files

- `hyperopt/scripts/common.py`
- `scripts/dataset_conversion/README.md`
- `docs/guides/experiment_guide.md`
- `src/training/README.md`
