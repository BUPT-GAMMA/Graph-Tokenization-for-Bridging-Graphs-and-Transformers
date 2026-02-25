# TokenizerGraph

[[中文文档 / Chinese README]](README_zh.md)

> **Branches:** This is the **`release`** branch — clean code for reproducing paper experiments.
> For the full development version (utility scripts, benchmarks, internal docs), switch to the **[`dev` branch](../../tree/dev)**.

TokenizerGraph is a framework for molecular property prediction through graph serialization. It converts molecular graphs into token sequences, applies BPE compression to discover substructure patterns, and uses Transformer encoders (BERT or GTE) for pre-training and fine-tuning on downstream tasks.

## Overview

The core idea is straightforward: serialize a molecular graph into a linear token sequence, then treat it as a "sentence" that a language model can learn from. Different serialization strategies (Eulerian paths, DFS, BFS, etc.) capture different structural aspects of the graph. BPE compression then merges frequent token pairs into higher-level tokens, effectively learning common substructures.

```
Raw Molecules → Graph Construction → Serialization → BPE Compression → Transformer → Predictions
```

## Project Structure

```
tokenizerGraph/
├── prepare_data_new.py         # Data preprocessing: serialization + BPE training + vocab
├── run_pretrain.py             # Pre-training entry point (MLM)
├── run_finetune.py             # Fine-tuning entry point (regression/classification)
├── batch_pretrain_simple.py    # Batch pre-training across datasets/methods/GPUs
├── batch_finetune_simple.py    # Batch fine-tuning
├── aggregate_results.py        # Collect and tabulate experiment results
├── config.py                   # Centralized configuration management
├── config/default_config.yml   # Default config values
├── src/
│   ├── algorithms/
│   │   ├── serializer/         # Graph serialization (Euler, DFS, BFS, Topo, SMILES, ...)
│   │   └── compression/        # BPE engine (C++ / Numba / Python backends)
│   ├── data/                   # Data loading and preprocessing
│   │   └── loader/             # Per-dataset loaders
│   ├── models/                 # Model definitions
│   │   ├── bert/               # BERT encoder
│   │   ├── gte/                # GTE encoder (Alibaba-NLP/gte-multilingual-base)
│   │   └── unified_encoder.py  # Unified encoder interface
│   ├── training/               # Training pipelines and utilities
│   └── utils/                  # Logging, metrics, visualization
├── gte_model/                  # Local GTE model config (for offline use)
├── final/                      # Paper experiment scripts and plotting code
└── docs/                       # Documentation
```

## Installation

```bash
git clone <repository_url>
cd TokenizerGraph

# Install in development mode
pip install -e .

# Build the C++ BPE backend (optional but recommended for speed)
python setup.py build_ext --inplace
```

Key dependencies: `torch`, `dgl`, `networkx`, `rdkit`, `transformers`, `pybind11`, `pandas`.

## Usage

### 1. Data Preparation

Serialize molecular graphs and train a BPE tokenizer:

```bash
python prepare_data_new.py \
    --datasets qm9test \
    --methods feuler \
    --bpe_merges 2000
```

This loads the dataset, serializes all graphs with the chosen method, trains a BPE model on the resulting sequences, and builds a vocabulary. All artifacts are cached for reuse.

### 2. Pre-training

Pre-train a Transformer encoder with Masked Language Modeling (MLM):

```bash
python run_pretrain.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --epochs 100 \
    --batch_size 256
```

### 3. Fine-tuning

Fine-tune the pre-trained model on a downstream property prediction task:

```bash
python run_finetune.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --target_property homo \
    --epochs 200 \
    --batch_size 64
```

### 4. Batch Experiments

Run experiments across multiple datasets, serialization methods, and GPUs in parallel:

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

## Reproducing Paper Experiments

Scripts for all paper experiments are in the `final/` directory:

- **Main experiments** — `final/exp1_main/run/`: pre-training and fine-tuning commands
- **Efficiency analysis** — `final/exp1_speed/`: serialization speed, token length stats, training throughput
- **Multi-sampling comparison** — `final/exp2_mult_seralize_comp/`: effect of multiple serialization samples
- **BPE vocabulary visualization** — `final/exp4_bpe_vocab_visual/`: codebook inspection and visualization

## Documentation

- [Configuration Guide](docs/guides/config_guide.md) — config file structure and parameters
- [Experiment Guide](docs/guides/experiment_guide.md) — how to design and run experiments
- [BPE Usage Guide](docs/bpe/BPE_USAGE_GUIDE.md) — BPE engine API and usage

## Branches

- **`release`** — Clean version with only the code needed to reproduce paper experiments.
- **`dev`** — Full development version with all utility scripts, benchmarks, and internal documentation.
