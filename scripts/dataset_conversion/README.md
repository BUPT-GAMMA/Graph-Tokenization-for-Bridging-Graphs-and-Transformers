# Dataset Conversion Scripts

Scripts for inspecting raw graph datasets and converting them into the unified format expected by TokenizerGraph.

## Overview

TokenizerGraph expects each dataset to be stored as:

```
data/<dataset>/
├── data.pkl              # List of (dgl_graph, label) tuples or List[Dict]
├── train_index.json      # Train split indices
├── val_index.json        # Validation split indices
└── test_index.json       # Test split indices
```

All graphs must have:
- `g.ndata['feat']` — node token IDs (`LongTensor`, shape `[N, 1]` or `[N, 2]`)
- `g.edata['feat']` — edge token IDs (`LongTensor`, shape `[E, 1]`)

This directory layout is only the raw-data prerequisite for the main training pipeline. A dataset should be treated as training-ready only when all of the following are true:

1. `data/<dataset>/data.pkl` exists
2. `data/<dataset>/train_index.json`, `val_index.json`, `test_index.json` exist
3. the dataset name is registered in `src/data/unified_data_factory.py`
4. `python prepare_data_new.py --datasets <dataset> --methods feuler` finishes successfully
5. `data/processed/<dataset>/...` artifacts are created

## Audited Dataset Availability

The audited status below reflects the checked-in repository state on 2026-03-15. The classification is intentionally strict:

- `Loader OK` means the dataset is registered in `src/data/unified_data_factory.py` and `loader.load_data()` succeeds with the local files currently present in `data/<dataset>/`.
- `feuler training cache readable` means `UnifiedDataInterface` can read the existing `feuler` serialized cache, vocab, and BPE codebook from the current repository state.
- `End-to-end verified` is reserved for datasets that were actually executed through `prepare_data_new.py -> run_pretrain.py -> run_finetune.py`.
- `feuler training cache readable` does not imply that the dataset has been reproduced from raw data in the current audit window.

| Status | Datasets | Meaning |
| --- | --- | --- |
| End-to-end verified | `qm9test` | `prepare_data_new.py -> run_pretrain.py -> run_finetune.py` has been executed successfully in the current repository state. |
| Loader OK + feuler training cache readable | `aqsol`, `coildel`, `colors3`, `dblp`, `dd`, `molhiv`, `mutagenicity`, `peptides_func`, `peptides_struct`, `proteins`, `qm9`, `qm9test`, `synthetic`, `twitter`, `zinc` | Raw data files load successfully, and the current `feuler` cache can be consumed by the training read path. |
| Loader OK, prepare required before training | `mnist`, `mnist_raw` | Loader smoke tests succeed, but the current repository state does not contain the required `feuler` serialized cache, vocab, and BPE codebook. |
| Blocked in current repository state | `code2` | The raw loader path is blocked because `data/code2/data.pkl` is missing. Existing partial cache files are not sufficient to claim that the dataset is runnable. |

## Getting Datasets

### Option A: Download Pre-processed Data

Pre-processed datasets (in the format above) can be downloaded from:

> **TODO**: Add release download link (e.g., Zenodo, Google Drive, or GitHub Releases).

Extract into the `data/` directory at the project root.

### Option B: Convert from Raw Sources

1. **Install dependencies**: `pip install dgl ogb torch_geometric rdkit`

2. **Inspect raw datasets** (optional, to verify fields):
   ```bash
   # Inspect DGL/TU datasets
   python scripts/dataset_conversion/check_dgl_graphpred.py --datasets PROTEINS COLORS-3

   # Inspect OGB datasets
   python scripts/dataset_conversion/check_ogbg.py --datasets ogbg-molhiv
   ```

3. **Run conversion**: Each dataset loader in `src/data/loader/` reads from `data/<dataset>/data.pkl`. To generate these files from raw sources, use the appropriate conversion approach below.

4. **Verify final directory layout** before running any training command:

   ```text
   data/<dataset>/
   ├── data.pkl
   ├── train_index.json
   ├── val_index.json
   └── test_index.json
   ```

5. **Run preprocessing** to build serialized sequences and vocabularies:

   ```bash
   python prepare_data_new.py --datasets <dataset> --methods feuler
   ```

If multi-sampling is used during preparation, the exact `K` value must be preserved. Training must use the same `serialization.multiple_sampling.num_realizations=K`; otherwise it will read from a different cache directory.

6. **Run smoke tests**:

   ```bash
   python run_pretrain.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
   python run_finetune.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
   ```

   For fine-tuning, CUDA must be available and the command should point to a valid pre-trained checkpoint directory.

## Dataset Sources & Conversion

### Molecular Datasets (QM9, ZINC, AQSOL)

| Dataset | Source | Task | Raw Format |
|---------|--------|------|------------|
| `qm9` | [MoleculeNet](https://moleculenet.org/) / DGL built-in | Regression (16 properties) | DGL graphs with atom/bond features |
| `zinc` | [ZINC-12K](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin#zinc) | Regression (logP) | DGL graphs |
| `aqsol` | [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) | Regression (solubility) | DGL graphs |

Node tokens = atomic number; Edge tokens = bond type (SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4).

Expected practical notes:

- `qm9` / `qm9test` loaders expect `data.pkl` and split JSON files
- they may also read optional SMILES side files if present
- if SMILES files are absent, the core graph pipeline can still run as long as the required files exist

### OGB Datasets (MolHIV, Peptides)

| Dataset | Source | Task | Notes |
|---------|--------|------|-------|
| `molhiv` | [OGB](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) `ogbg-molhiv` | Binary classification | Official train/val/test splits |
| `peptides_func` | [LRGB](https://github.com/vijaydwivedi75/lrgb) | Multi-label classification (10 classes) | Official splits |
| `peptides_struct` | [LRGB](https://github.com/vijaydwivedi75/lrgb) | Multi-target regression | Official splits |

### TU Datasets (PROTEINS, COLORS-3, DD, etc.)

| Dataset | Source | Task | Node Token Source |
|---------|--------|------|-------------------|
| `proteins` | [TU](https://chrsmrrs.github.io/datasets/) `PROTEINS` | Binary classification | `node_labels` |
| `colors3` | TU `COLORS-3` | 11-class classification | `node_attr` (one-hot → discrete ID) |
| `synthetic` | TU `SYNTHETIC` | Binary classification | `node_labels` |
| `mutagenicity` | TU `Mutagenicity` | Binary classification | `node_labels` + `edge_labels` |
| `dd` | TU `DD` | Binary classification | `node_labels` |
| `coildel` | TU `COIL-DEL` | 100-class classification | 2-channel `node_attr` |
| `dblp` | TU `DBLP_v1` | Binary classification | `node_labels` |
| `twitter` | TU `TWITTER-Real-Graph-Partial` | Binary classification | `node_labels` |

**Token domain separation**: Node tokens are mapped to odd integers (`2x+1`), edge tokens to even integers (`2x`). See `DGL_tokenization_prep_plan.md` for details.

### Special Datasets

| Dataset | Source | Task |
|---------|--------|------|
| `mnist` | MNIST superpixel graphs | 10-class classification |
| `code2` | OGB `ogbg-code2` | Code AST classification |

## Scripts in This Directory

| File | Purpose |
|------|---------|
| `check_dgl_graphpred.py` | Inspect DGL/TU graph datasets: node/edge features, label distribution, graph statistics |
| `check_ogbg.py` | Inspect OGB graph property prediction datasets |
| `DGL_tokenization_prep_plan.md` | Detailed tokenization specification for each dataset (Chinese, internal reference) |
| `DGL_graph_pred_tokenizable_nodes.md` | Survey of DGL datasets with discrete node features suitable for tokenization |

## After Conversion

Once `data/<dataset>/` contains the required files, run the full pipeline:

```bash
# Step 1: Serialize + BPE + build vocab
python prepare_data_new.py --datasets qm9test --methods feuler

# Step 2: Pre-train
python run_pretrain.py --dataset qm9test --method feuler

# Step 3: Fine-tune
python run_finetune.py --dataset qm9test --method feuler
```

Note the CLI difference:

- `prepare_data_new.py` uses plural arguments: `--datasets`, `--methods`
- `run_pretrain.py` / `run_finetune.py` use singular arguments: `--dataset`, `--method`

If preparation uses `--multiple_samples K`, the training scripts must be launched with matching `serialization.multiple_sampling.enabled=true` and `serialization.multiple_sampling.num_realizations=K`. Otherwise `UnifiedDataInterface` reads from `single/` instead of `multi_<K>/`.

Skipping Step 1 leaves the serialized cache and vocabulary absent, so later training commands will fail.

## Minimal End-to-End Validation

The following `qm9test` sequence was executed successfully on the current repository state. It is the only dataset/configuration pair that has been fully verified through `prepare -> pretrain -> finetune` in this audit.

1. Prepare a fresh `multi_3` cache to avoid overwriting the existing `single/`, `multi_10/`, or `multi_100/` artifacts:

    ```bash
    python prepare_data_new.py \
        --datasets qm9test \
        --methods feuler \
        --multiple_samples 3 \
        --workers 1 \
        --bpe_merges 64 \
        --bpe_min_freq 2 \
        --out prepare_results/e2e_qm9test_feuler_multi3
    ```

    Expected new artifacts:

    ```text
    data/processed/qm9test/serialized_data/feuler/multi_3/serialized_data.pickle
    data/processed/qm9test/vocab/feuler/bpe/multi_3/vocab.json
    model/bpe/qm9test/feuler/multi_3/bpe_codebook.pkl
    ```

2. Run a one-epoch pre-training smoke test against the same `multi_3` cache:

    ```bash
    CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
        --dataset qm9test \
        --method feuler \
        --experiment_group e2e_audit \
        --experiment_name e2e_qm9test_feuler_multi3_pretrain \
        --epochs 1 \
        --batch_size 256 \
        --learning_rate 1e-4 \
        --bpe_encode_rank_mode all \
        --log_style offline \
        --plain_logs \
        --config_json '{"device":"cuda:0","system":{"device":"cuda:0","num_workers":1,"persistent_workers":true},"serialization":{"multiple_sampling":{"enabled":true,"num_realizations":3},"bpe":{"num_merges":64}}}'
    ```

    This creates a checkpoint at:

    ```text
    model/e2e_audit/e2e_qm9test_feuler_multi3_pretrain/run_0/best/
    ```

    The directory must contain both `config.bin` and `pytorch_model.bin`.

3. Run a one-epoch fine-tuning smoke test using that explicit checkpoint directory:

    ```bash
    CUDA_VISIBLE_DEVICES=1 python run_finetune.py \
        --dataset qm9test \
        --method feuler \
        --target_property homo \
        --pretrained_dir model/e2e_audit/e2e_qm9test_feuler_multi3_pretrain/run_0/best \
        --experiment_group e2e_audit \
        --experiment_name e2e_qm9test_feuler_multi3_finetune \
        --epochs 1 \
        --batch_size 256 \
        --learning_rate 1e-5 \
        --bpe_encode_rank_mode all \
        --aggregation_mode avg \
        --log_style offline \
        --plain_logs \
        --config_json '{"device":"cuda:0","system":{"device":"cuda:0","num_workers":1,"persistent_workers":true},"serialization":{"multiple_sampling":{"enabled":true,"num_realizations":3},"bpe":{"num_merges":64}},"bert":{"finetuning":{"save_models":false}}}'
    ```

If all three steps pass, the `prepare_data_new.py -> run_pretrain.py -> run_finetune.py` path is confirmed for that dataset/configuration pair only.

## Current Repository Caveats

- `dataset.limit` should be treated as legacy metadata rather than a reliable way to subset data for smoke tests. For minimal verification, use `qm9test` or prepare a dedicated smaller dataset.
- The checked-in default config uses `encoder.type: gte`, so a run will use the GTE encoder unless you explicitly switch to `bert`.
- `run_finetune.py` currently asserts that CUDA is available before any training starts.
- In the checked-in repository state audited here, `code2` does not pass the loader smoke test because `data/code2/data.pkl` is missing, so it should not be documented as immediately runnable without regenerating its processed data first.
- A separate cold-start reproducibility audit is maintained in:
  - `docs/reproducibility/dataset-cold-start-audit.md`
  - `docs/reproducibility/cold-start-runbook.md`

## Adding a New Dataset

1. Write a conversion script that produces `data/<name>/data.pkl` + split indices
2. Create a loader class in `src/data/loader/<name>_loader.py` inheriting from `BaseDataLoader`
3. Register it in `src/data/unified_data_factory.py`
4. See `DGL_tokenization_prep_plan.md` for the token mapping convention
