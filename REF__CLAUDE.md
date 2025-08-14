# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
TokenizerGraph is a research project for converting graph-structured data (molecules, networks) into sequences for transformer models like BERT. It implements 7 different graph serialization algorithms and compares their effectiveness.

## Key Commands

### Data Preparation
```bash
python data_prepare.py --dataset qm9 --subgraph_limit 1000
python data_prepare.py --dataset qm9 --serialization_method graph_seq
python data_prepare.py --dataset qm9 --bpe_num_merges 2000
```

### Training
```bash
# BERT pre-training
python bert_pretraining_pipeline_optimized.py

# Parallel training
python parallel_bert_pretraining.py --dataset qm9 --gpu_count 4

# Fine-tuning
python bert_finetuning_pipeline_normalized.py
```

### Evaluation
```bash
# Compare all algorithms
python run_serialization_bpe_comparison_simple.py --dataset qm9

# Run tests
python -m pytest tests/
```

## Architecture
- **Serialization**: 7 algorithms (graph_seq, dfs, bfs, eulerian, topological, smiles, cpp)
- **Data**: QM9 molecules, citation networks, social networks via DGL
- **Models**: BERT for sequences, GNN for baselines
- **Compression**: BPE for sequence compression

## Key Files
- `config/default_config.yml` - Main configuration
- `src/algorithms/serializer/` - All 7 serialization algorithms
- `src/data/qm9_loader.py` - QM9 dataset handling
- `bert_pretraining_pipeline_optimized.py` - Main training pipeline

## Development
Use `config.py` for unified configuration management. All serialization algorithms implement the same interface in `src/algorithms/serializer/`.