# Hyperparameter Search

This directory contains the maintained Optuna-based hyperparameter search workflow for TokenizerGraph.

## Framework

- Search backend: `Optuna`
- Shared storage: `JournalStorage` with `JournalFileBackend`
- Search stages:
  - `hyperopt/scripts/large_batch_search.py` for pretraining search and optional finetuning search from top pretrain trials
  - `hyperopt/scripts/extract_best_params_for_finetuning.py` for exporting reusable pretrain candidates
  - `hyperopt/scripts/finetune_with_pretrain_options.py` for finetuning search over exported pretrain options
  - `hyperopt/scripts/analyze_optuna_results.py` for study summaries

## Layout

```text
hyperopt/
├── README.md
├── journal/                  # Optuna journal files (not versioned)
├── results/                  # Exported option JSON files (not versioned)
└── scripts/
    ├── common.py
    ├── large_batch_search.py
    ├── extract_best_params_for_finetuning.py
    ├── finetune_with_pretrain_options.py
    └── analyze_optuna_results.py
```

## Quick Start

1. Run pretrain search:

```bash
python hyperopt/scripts/large_batch_search.py \
  --dataset zinc \
  --methods fcpp \
  --bpe_mode all \
  --bpe_backend cpp \
  --stage pretrain \
  --pretrain_trials 20
```

2. Export the top pretrain checkpoints:

```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/large_batch.db \
  --dataset zinc \
  --bpe_mode all \
  --output_dir hyperopt/results
```

3. Run finetune search over the exported candidates:

```bash
python hyperopt/scripts/finetune_with_pretrain_options.py \
  --dataset zinc \
  --bpe_mode all \
  --bpe_backend cpp \
  --options_file hyperopt/results/best_pretrain_params_for_finetuning.json \
  --trials 20
```

4. Inspect the resulting studies:

```bash
python hyperopt/scripts/analyze_optuna_results.py \
  --journal hyperopt/journal/large_batch.db \
  --dataset zinc \
  --bpe_mode all \
  --stage pretrain
```

## Notes

- The maintained workflow uses the in-repo training pipelines directly, so Optuna pruning remains available.
- `--bpe_backend cpp` is the default for real runs and requires `python setup.py build_ext --inplace`.
- `--bpe_backend python` is available as a fallback for environments without `_cpp_bpe`, but it is intended for deterministic smoke/debug runs and currently supports `--bpe_mode all|none`.
- Output files under `hyperopt/journal/` and `hyperopt/results/` are run artifacts and should not be committed.
- For a full reproducibility guide, see `docs/guides/hyperparameter_search.md`.
