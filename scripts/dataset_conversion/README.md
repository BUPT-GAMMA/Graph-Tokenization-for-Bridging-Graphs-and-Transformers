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

## Dataset Sources & Conversion

### Molecular Datasets (QM9, ZINC, AQSOL)

| Dataset | Source | Task | Raw Format |
|---------|--------|------|------------|
| `qm9` | [MoleculeNet](https://moleculenet.org/) / DGL built-in | Regression (16 properties) | DGL graphs with atom/bond features |
| `zinc` | [ZINC-12K](https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin#zinc) | Regression (logP) | DGL graphs |
| `aqsol` | [AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) | Regression (solubility) | DGL graphs |

Node tokens = atomic number; Edge tokens = bond type (SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4).

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
python prepare_data_new.py --dataset qm9test --method feuler

# Step 2: Pre-train
python run_pretrain.py --dataset qm9test --method feuler

# Step 3: Fine-tune
python run_finetune.py --dataset qm9test --method feuler
```

## Adding a New Dataset

1. Write a conversion script that produces `data/<name>/data.pkl` + split indices
2. Create a loader class in `src/data/loader/<name>_loader.py` inheriting from `BaseDataLoader`
3. Register it in `src/data/unified_data_factory.py`
4. See `DGL_tokenization_prep_plan.md` for the token mapping convention
