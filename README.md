# GraphTokenizer

**Graph Tokenization for Bridging Graphs and Transformers**

[[中文文档 / Chinese README]](README_zh.md) · [[Paper (ICLR 2026 / OpenReview)]](https://openreview.net/forum?id=jCctxI1BGF) · arXiv (coming soon)

> **Branches:** `release` — clean code for reproducing paper experiments. [`dev`](../../tree/dev) — full development version with utility scripts, benchmarks, and internal docs.

## Overview

The success of large pretrained Transformers is closely tied to tokenizers, which convert raw input into discrete symbols. **GraphTokenizer** extends this paradigm to graph-structured data by introducing a general **graph tokenization** framework. It converts arbitrary labeled graphs into discrete token sequences, enabling standard off-the-shelf Transformer models (e.g., BERT, GTE) to be applied directly to graph data **without any architectural modifications**.

The framework combines **reversible graph serialization** with **Byte Pair Encoding (BPE)**, the de facto standard tokenizer in large language models. To better capture structural information, the serialization process is guided by **global statistics of graph substructures**, ensuring that frequently occurring substructures appear as adjacent symbols in the resulting sequence — an ideal input for BPE to discover a meaningful vocabulary of structural graph tokens. The entire process is **reversible**: the original graph can be faithfully reconstructed from its token sequence.

<p align="center">
  <img src="docs/assets/framework.jpg" width="90%" alt="GraphTokenizer Framework">
</p>

**Framework overview.** **(A)** Substructure frequencies (labeled-edge patterns) are collected from the training graphs. **(B)** Structure-guided reversible serialization via frequency-guided Eulerian circuit — at each node, the next edge is selected according to the frequency priority (e.g., at the red C node, the C–C pattern has the highest frequency, so that edge is traversed first). **(C)** A BPE vocabulary is trained on the serialized corpus; BPE iteratively merges the most frequent adjacent symbol pairs into new tokens, compressing sequences to ~10% of their original length while preserving common substructures.

```
Labeled Graphs  →  Structure-Guided Serialization  →  BPE Tokenization  →  Transformer  →  Predictions
```

### Key Contributions

- **General Graph Tokenization Framework.** Combines reversible graph serialization with BPE to create a bidirectional interface between graphs and sequence models. By decoupling the encoding of graph structure from the model architecture, it enables standard off-the-shelf Transformers to process graph data without any architectural modifications.
- **Structure-Guided Serialization for BPE.** A deterministic serialization mechanism guided by global substructure statistics. It addresses the ordering ambiguity inherent in graphs (permutation invariance) and systematically arranges frequent substructures into adjacent symbol patterns — precisely the input that BPE's greedy merging strategy is designed to exploit.
- **State-of-the-Art on 14 Benchmarks.** Achieves SOTA results across diverse graph classification and regression benchmarks spanning molecular, biomedical, social, academic, and synthetic domains. Scaling from a compact BERT-small to a larger GTE backbone yields consistent gains, demonstrating that graph tokenization can leverage the proven scaling behavior of Transformers.

### Main Results

Classification (↑ higher is better) and regression (↓ lower is better) results:

| Model | molhiv (AUC↑) | p-func (AP↑) | mutag (Acc↑) | coildel (Acc↑) | dblp (Acc↑) | qm9 (MAE↓) | zinc (MAE↓) | aqsol (MAE↓) | p-struct (MAE↓) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCN | 74.0 | 53.2 | 79.7 | 74.6 | 76.6 | 0.134 | 0.399 | 1.345 | 0.342 |
| GIN | 76.1 | 61.4 | 80.4 | 72.0 | 73.8 | 0.176 | 0.379 | 2.053 | 0.338 |
| GAT | 72.1 | 51.2 | 80.1 | 74.4 | 76.3 | 0.114 | 0.445 | 1.388 | 0.316 |
| GatedGCN | 80.6 | 51.2 | 83.6 | 83.7 | 86.0 | 0.096 | 0.370 | 0.940 | 0.312 |
| GraphGPS | 78.5 | 53.5 | 84.3 | 80.5 | 71.6 | 0.084 | 0.310 | 1.587 | 0.251 |
| Exphormer | 82.3 | 64.5 | 82.7 | **91.5** | 84.9 | 0.080 | 0.281 | 0.749 | 0.251 |
| GraphMamba | 81.2 | 67.7 | 85.0 | 74.5 | 87.6 | 0.083 | 0.209 | 1.133 | 0.248 |
| GCN+ | 80.1 | 72.6 | 88.7 | 88.9 | 89.6 | 0.077 | **0.116** | 0.712 | 0.244 |
| **GT+BERT** | 82.6 | 68.5 | 87.5 | 74.1 | 93.2 | 0.122 | 0.241 | 0.648 | 0.247 |
| **GT+GTE** | **87.4** | **73.1** | **90.1** | 89.6 | **93.6** | **0.071** | 0.131 | **0.609** | **0.242** |

Results are mean over 5 independent runs. **Bold** = best. See the paper for full results on all 14 datasets including DD, Twitter, Proteins, Colors-3, and Synthetic.

### Supported Serialization Methods

| Method | Reversible | Deterministic | Applicable to |
|:---|:---:|:---:|:---|
| Freq-Guided Eulerian (Feuler) | ✅ | ✅ | Any labeled graph |
| Freq-Guided CPP (FCPP) | ✅ | ✅ | Any labeled graph |
| Eulerian circuit | ✅ | ❌ | Any labeled graph |
| Chinese Postman (CPP) | ✅ | ❌ | Any labeled graph |
| Canonical SMILES | ✅ | ✅ | Molecular graphs only |
| DFS / BFS / Topo | ❌ | ❌ | Any graph |

The default method is **Feuler** (Frequency-Guided Eulerian circuit), which provides both reversibility and determinism with O(|E|) time complexity.

## Project Structure

```
GraphTokenizer/
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
│   │   ├── serializer/         # Graph serialization (Freq-Euler, Euler, DFS, BFS, Topo, SMILES, CPP, ...)
│   │   └── compression/        # BPE engine (C++ / Numba / Python backends)
│   ├── data/                   # Unified data interface and per-dataset loaders
│   │   └── loader/             # Per-dataset loaders (QM9, ZINC, AQSOL, MNIST, Peptides, ...)
│   ├── models/                 # Model definitions
│   │   ├── bert/               # BERT encoder, vocab manager, data pipeline
│   │   ├── gte/                # GTE encoder (Alibaba-NLP/gte-multilingual-base)
│   │   └── unified_encoder.py  # Unified encoder interface
│   ├── training/               # Training pipelines (pretrain, finetune, evaluation)
│   └── utils/                  # Logging, metrics, visualization
├── gte_model/                  # Local GTE model config (for offline use)
├── final/                      # Paper experiment scripts and plotting code
└── docs/                       # Documentation
```

## Installation

```bash
git clone https://github.com/BUPT-GAMMA/GraphTokenizer.git
cd GraphTokenizer

# Install in development mode
pip install -e .

# Build the C++ BPE backend (optional but recommended for speed)
python setup.py build_ext --inplace
```

Key dependencies: `torch`, `dgl`, `networkx`, `rdkit`, `transformers`, `pybind11`, `pandas`.

## Quick Start

### 1. Data Preparation

Before running `prepare_data_new.py`, make sure the raw/preprocessed dataset files already exist under `data/<dataset>/`.

The loaders in `src/data/loader/` assume the following files are present:

```text
data/<dataset>/
├── data.pkl
├── train_index.json
├── val_index.json
└── test_index.json
```

For molecular datasets such as `qm9` and `zinc`, some loaders will also look for optional SMILES files such as `smiles_1_direct.txt`.

If you are using the repository exactly as released by the authors, the simplest smoke test is to start from `qm9test`, because it is the smallest built-in example used throughout the codebase.

Serialize graphs and train a BPE tokenizer:

```bash
python prepare_data_new.py \
    --datasets qm9test \
    --methods feuler \
    --bpe_merges 2000
```

This script:

- loads `data/qm9test/data.pkl` together with the fixed split files
- serializes every graph with the selected method
- trains a BPE model on the serialized corpus
- builds the vocabulary used by downstream Transformer runs
- writes cached artifacts under `data/processed/<dataset>/...`

After this step, you should expect processed artifacts in locations similar to:

```text
data/processed/qm9test/
├── serialized_data/feuler/single/serialized_data.pickle
└── vocab/feuler/bpe/single/vocab.json
```

Refer to the following resources for detailed data preparation and execution instructions:

- [`scripts/dataset_conversion/README.md`](scripts/dataset_conversion/README.md) — dataset-by-dataset conversion notes
- [`src/data/README.md`](src/data/README.md) — data layer contract and expected directory layout
- [`docs/reproducibility/dataset-cold-start-audit.md`](docs/reproducibility/dataset-cold-start-audit.md) — cold-start reproducibility audit and script traceability
- [`docs/reproducibility/cold-start-runbook.md`](docs/reproducibility/cold-start-runbook.md) — independent clone-based cold-start run record
- [`docs/reproducibility/cold-start-roadmap.md`](docs/reproducibility/cold-start-roadmap.md) — remaining dataset-by-dataset closure plan
- [`docs/reproducibility/paper-dataset-cold-start-guide.md`](docs/reproducibility/paper-dataset-cold-start-guide.md) — formal paper-scope dataset setup and validation guide

Current audited status:

- `qm9test` is the only dataset that has been fully verified through `prepare_data_new.py -> run_pretrain.py -> run_finetune.py`
- `mnist` and `mnist_raw` currently pass loader-level checks only; `prepare_data_new.py` must be executed before training
- `code2` is blocked in the checked-in repository state because `data/code2/data.pkl` is missing
- The complete audited status table is maintained in [`scripts/dataset_conversion/README.md`](scripts/dataset_conversion/README.md)

**Important Notes:**

- `prepare_data_new.py` uses plural CLI arguments (`--datasets`, `--methods`), while `run_pretrain.py` and `run_finetune.py` use singular ones (`--dataset`, `--method`)
- If preparing data with `--multiple_samples K`, the training scripts must be launched with matching `serialization.multiple_sampling.enabled=true` and `serialization.multiple_sampling.num_realizations=K`; otherwise, they will read from `single/` instead of `multi_K/`
- The checked-in default config currently sets `encoder.type: gte`, so runs will use the GTE encoder unless explicitly switched to `bert`

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

**Important Notes:**

- `--dataset` and `--method` are required
- The script reads the processed artifacts produced by `prepare_data_new.py`
- The default config uses the paths in `config/default_config.yml`, where `data_dir` resolves to `data/`
- A verified one-epoch `qm9test` smoke test with `multi_3` serialization is documented in [`scripts/dataset_conversion/README.md`](scripts/dataset_conversion/README.md)

### 3. Fine-tuning

Fine-tune the pre-trained model on downstream graph prediction tasks:

```bash
python run_finetune.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --target_property homo \
    --epochs 200 \
    --batch_size 64
```

For regression datasets such as `qm9`, set `--target_property` explicitly. For classification datasets such as `mutagenicity` or `molhiv`, the loader metadata is usually sufficient and no regression target is needed.

**Fine-tuning Notes:**

- `run_finetune.py` currently requires CUDA (`torch.cuda.is_available()` is asserted at startup)
- For smoke tests, `--pretrained_dir` should point directly to `model/<group>/<exp_name>/run_0/best`
- The pre-trained checkpoint directory must contain both `config.bin` and `pytorch_model.bin`

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

- **Main experiments** — `final/exp1_main/run/`: pre-training and fine-tuning commands for all 14 datasets
- **Efficiency analysis** — `final/exp1_speed/`: serialization speed, token length stats, training throughput
- **Multi-sampling comparison** — `final/exp2_mult_seralize_comp/`: effect of multiple serialization samples
- **BPE vocabulary visualization** — `final/exp4_bpe_vocab_visual/`: codebook inspection and visualization

## Dataset Preparation Checklist

Use the following checklist to verify that a dataset is runnable end-to-end:

1. Put the dataset under `data/<dataset>/`
2. Ensure `data.pkl`, `train_index.json`, `val_index.json`, and `test_index.json` all exist
3. Confirm the dataset name is registered in `src/data/unified_data_factory.py`
4. Run:

```bash
python prepare_data_new.py --datasets <dataset> --methods feuler
```

5. Verify that `data/processed/<dataset>/serialized_data/...` and `data/processed/<dataset>/vocab/...` were created
6. If you prepared with multiple sampling, also verify whether the artifacts were written to `single/` or `multi_<K>/`
7. Run a small pre-training smoke test:

```bash
python run_pretrain.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
```

8. Run a small fine-tuning smoke test:

```bash
python run_finetune.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
```

Fine-tuning also requires a CUDA-capable device and a valid pre-trained checkpoint.

## Documentation

- [Configuration Guide](docs/guides/config_guide.md) — config file structure and parameters
- [Experiment Guide](docs/guides/experiment_guide.md) — how to design and run experiments
- [BPE Usage Guide](docs/bpe/BPE_USAGE_GUIDE.md) — BPE engine API and usage
- [Dataset Conversion Guide](scripts/dataset_conversion/README.md) — how to prepare `data/<dataset>/` so the loaders can run directly

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{guo2026graphtokenizer,
  title={Graph Tokenization for Bridging Graphs and Transformers},
  author={Guo, Zeyuan and Diao, Enmao and Yang, Cheng and Shi, Chuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## Branches

- **`release`** — Clean version with only the code needed to reproduce paper experiments.
- **`dev`** — Full development version with all utility scripts, benchmarks, and internal documentation.
