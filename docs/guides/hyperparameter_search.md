# Hyperparameter Search Guide

This guide documents the maintained hyperparameter search workflow for TokenizerGraph.

## What We Use

The project uses `Optuna` as the search framework. The maintained scripts use:

- `TPESampler` for parameter suggestion
- `MedianPruner` or `PercentilePruner` for early stopping
- `JournalStorage` + `JournalFileBackend` so multiple processes can share a file-backed study

Official reference:

- Optuna documentation: <https://optuna.readthedocs.io/en/stable/>
- Journal storage reference: <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html>

## Maintained Entry Points

- `hyperopt/scripts/large_batch_search.py`
  - Runs pretraining search.
  - Can also run finetuning search directly from the top-K completed pretraining trials.
- `hyperopt/scripts/extract_best_params_for_finetuning.py`
  - Exports the best pretraining trials to `hyperopt/results/best_pretrain_params_for_finetuning.json`.
- `hyperopt/scripts/finetune_with_pretrain_options.py`
  - Runs finetuning search against the exported pretraining candidates.
- `hyperopt/scripts/analyze_optuna_results.py`
  - Prints a compact summary of the study contents and best trials.

## Prerequisites

Before running search, make sure the dataset is already training-ready:

1. Raw data or the released bundle is available under `data/<dataset>/`.
2. `prepare_data_new.py` has already produced the required serialized cache and vocabulary for the dataset/method you want to search.
3. CUDA is available. The maintained search workflow fails fast if CUDA is unavailable.

Dataset setup references:

- `README.md`
- `README_zh.md`
- `scripts/dataset_conversion/README.md`

## Recommended Workflow

### 1. Search pretraining hyperparameters

Example for a paper-scale dataset:

```bash
python hyperopt/scripts/large_batch_search.py \
  --dataset zinc \
  --methods fcpp \
  --bpe_mode all \
  --encoder gte \
  --stage pretrain \
  --batch_sizes 128,256,512 \
  --pretrain_epochs 50 \
  --pretrain_trials 20 \
  --journal_file hyperopt/journal/large_batch.db
```

Important arguments:

- `--dataset`: dataset name already prepared by `prepare_data_new.py`
- `--methods`: one or more serialization methods, comma-separated
- `--bpe_mode`: `none|all|topk|random|gaussian`
- `--bpe_backend`: `cpp` for the compiled backend, `python` for the pure-Python fallback
- `--encoder`: `bert` or `gte`
- `--batch_sizes`: search space for batch size; the script name keeps the historical “large batch” focus, but smaller values are also allowed for smoke tests
- `--config_json`: optional advanced override passed into `ProjectConfig`

Backend notes:

- `--bpe_backend cpp` is the default for normal runs and requires:

```bash
python setup.py build_ext --inplace
```

- `--bpe_backend python` is intended for cold-start smoke/debug runs when the C++ extension is unavailable.
- The current Python fallback supports deterministic encoding modes only: `--bpe_mode all` or `--bpe_mode none`.

Artifacts produced by pretrain search:

- Optuna journal file, e.g. `hyperopt/journal/large_batch.db`
- Pretrained checkpoints under `model/<experiment_group>/<experiment_name>/run_0/best/`

The experiment names are intentionally stable:

- Pretrain checkpoint example: `search_zinc_all_pt_003`
- Pretrain study example: `hyperopt_pretrain_zinc_all`

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

The JSON contains:

- overall top-K pretraining trials
- per-method top trials
- each candidate checkpoint path, experiment name, method, and raw Optuna parameters

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

For regression datasets that require a target property, pass it explicitly:

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

The finetuning search objective is normalized to Optuna’s `minimize` direction:

- minimizing metrics such as `mae` and `rmse` are used directly
- maximizing metrics such as `accuracy`, `roc_auc`, and `ap` are negated before returning the Optuna objective value

The raw metric value is still stored in Optuna `user_attrs`, so study inspection remains readable.

### 4. Inspect results

```bash
python hyperopt/scripts/analyze_optuna_results.py \
  --journal hyperopt/journal/large_batch.db \
  --dataset zinc \
  --bpe_mode all \
  --stage pretrain
```

## Smoke-Test Recipe

For a quick local validation with the repository’s audited smoke dataset, use `qm9test` and a small batch size:

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
  --journal_file hyperopt/journal/qm9test_smoke.db
```

Then:

```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/qm9test_smoke_python2.db \
  --dataset qm9test \
  --bpe_mode all \
  --output_dir hyperopt/results/qm9test_smoke

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
  --trials 1
```

Verified on the current repository state on 2026-03-21:

- pretrain smoke journal: `hyperopt/journal/qm9test_smoke_python2.db`
- pretrain checkpoint: `model/hyperopt_pretrain_qm9test_all/search_qm9test_all_pt_000/run_0/best/`
- exported options: `hyperopt/results/qm9test_smoke/best_pretrain_params_for_finetuning.json`
- finetune smoke journal: `hyperopt/journal/qm9test_finetune_smoke_python2.db`

The same smoke route also verified that:

- the maintained scripts can run end to end without `_cpp_bpe` when `--bpe_backend python` is selected
- `src/utils/info_display.py` now reports the correct pretrain/finetune hyperparameters in startup logs
- `src/training/finetune_pipeline.py` now writes `finetune_metrics.json` cleanly during Optuna-driven runs

## Related Project Docs

- `hyperopt/README.md`
- `scripts/dataset_conversion/README.md`
- `docs/guides/experiment_guide.md`
- `src/training/README.md`
