# Documentation

## Guides

- [Configuration Guide](guides/config_guide.md) — how the config system works, parameter reference, environment variables
- [Experiment Guide](guides/experiment_guide.md) — designing experiments, training workflow, evaluation
- [Hyperparameter Search Guide](guides/hyperparameter_search.md) — maintained Optuna workflow, study layout, and reproducible usage
- [Dataset Overview](datasets_overview.md) — dataset scope, metrics, storage format, and loader entry points
- [GTE Integration Guide](gte_integration_guide.md) — using project token IDs with the GTE encoder
- [Model Architecture](../model_ARCHITECTURE.md) — unified model construction and pretrained-weight loading flow

## BPE System

- [BPE Usage Guide](bpe/BPE_USAGE_GUIDE.md) — dynamic BPE compression: API, encoding modes, best practices
- [BPE Overview](bpe/README.md) — architecture overview and quick-start examples
- [BPE Transform Rework Notes](bpe/bpe_transform_rework.md) — retained design notes for online BPE transforms and multiple serialization

## Reproducibility Artifacts

- [Paper Dataset Cold-Start Guide](reproducibility/paper-dataset-cold-start-guide.md) — dataset preparation status and reproducibility notes
- [Final Experiment Loader](../final/README.md) — loading preserved experiment metrics into analysis tables

## Module-level Documentation

Each source module has its own README with implementation details:

- [Data Layer](../src/data/README.md) — dataset loaders, the Unified Data Interface (UDI)
- [Serialization](../src/algorithms/serializer/README.md) — graph-to-sequence algorithms
- [BPE Compression](../src/algorithms/compression/README.md) — BPE engine internals
- [Models](../src/models/README.md) — unified encoder interface, task heads, model factory
- [BERT Encoder](../src/models/bert/README.md) — BERT-specific components (config, vocab, datasets)
- [Training](../src/training/README.md) — pre-training, fine-tuning, evaluation pipelines
- [Dataset Conversion](../scripts/dataset_conversion/README.md) — raw dataset inspection and conversion scripts
