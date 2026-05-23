# BPE Compression System

The BPE (Byte Pair Encoding) module learns common token patterns from serialized graph sequences and compresses them into a compact vocabulary. This is the same idea behind subword tokenization in NLP, but applied to graph-derived token sequences.

For usage details and parameter reference, see the [BPE Usage Guide](BPE_USAGE_GUIDE.md).

## Quick Start

```python
from src.algorithms.compression.bpe_engine import BPEEngine

# Create engine (C++ backend recommended for speed)
engine = BPEEngine(
    train_backend="numba",
    encode_backend="cpp",
    encode_rank_mode="all",
)

# Train on token sequences
sequences = [[1, 2, 3, 4], [2, 3, 4, 5], ...]
engine.train(token_sequences=sequences, num_merges=2000, min_frequency=10)

# Encode
encoded = engine.encode([1, 2, 3, 4])
encoded_batch = engine.batch_encode(sequences)
```

## Encoding Modes

| Mode | Behavior | Typical Use |
|------|----------|-------------|
| `all` | Apply all learned merge rules | Default; deterministic, maximum compression |
| `topk` | Apply only the top-k most frequent rules | Good balance of compression and diversity |
| `random` | Randomly sample a subset of rules | Data augmentation during pre-training |
| `gaussian` | Sample rules with Gaussian-weighted probability | Smoother randomization, biased toward frequent rules |

## Backends

The engine supports multiple backends for training and encoding:

- **C++ backend** (`cpp`): Fastest option, recommended for production. Requires building via `python setup.py build_ext --inplace`.
- **Numba backend** (`numba`): Good CPU-accelerated alternative.
- **Python backend** (`python`): Pure Python fallback, no extra dependencies.

## Performance (QM9 dataset, 130K sequences)

- Training: ~27s, ~4,800 seq/s
- Encoding (C++ backend): ~195,000 seq/s
- Memory overhead: negligible
