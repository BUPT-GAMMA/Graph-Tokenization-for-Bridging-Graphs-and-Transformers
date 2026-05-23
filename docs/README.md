# Documentation

## Guides

- [Configuration Guide](guides/config_guide.md) — how the config system works, parameter reference, environment variables
- [Experiment Guide](guides/experiment_guide.md) — designing experiments, training workflow, evaluation
- [Hyperparameter Search Guide](guides/hyperparameter_search.md) — maintained Optuna workflow, study layout, and reproducible usage

## BPE System

- [BPE Usage Guide](bpe/BPE_USAGE_GUIDE.md) — dynamic BPE compression: API, encoding modes, best practices
- [BPE Overview](bpe/README.md) — architecture overview and quick-start examples

## Module-level Documentation

Each source module has its own README with implementation details:

- [Data Layer](../src/data/README.md) — dataset loaders, the Unified Data Interface (UDI)
- [Serialization](../src/algorithms/serializer/README.md) — graph-to-sequence algorithms
- [BPE Compression](../src/algorithms/compression/README.md) — BPE engine internals
- [Models](../src/models/README.md) — unified encoder interface, task heads, model factory
- [BERT Encoder](../src/models/bert/README.md) — BERT-specific components (config, vocab, datasets)
- [Training](../src/training/README.md) — pre-training, fine-tuning, evaluation pipelines
- [Dataset Conversion](../scripts/dataset_conversion/README.md) — raw dataset inspection and conversion scripts
