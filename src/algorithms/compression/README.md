# BPE Engine

Unified BPE (Byte Pair Encoding) engine for training merge rules and encoding token sequences. All access goes through `BPEEngine` in `bpe_engine.py`.

## Quick Start

```python
from src.algorithms.compression.bpe_engine import BPEEngine

engine = BPEEngine(train_backend='cpp', encode_backend='cpp')
engine.train(token_sequences, num_merges=2000, min_frequency=10)
engine.build_encoder()

encoded = engine.encode(seq)
encoded_batch = engine.batch_encode(seqs)
```

## Backends

| Component | Backend | Notes |
|-----------|---------|-------|
| Training | `cpp` | C++ minBPE implementation. Fastest. Requires `python setup.py build_ext --inplace`. |
| Training | `python` | Pure Python with incremental frequency table. Slower, slightly different semantics. |
| Encoding | `cpp` | C++/pybind11. Recommended. |
| Encoding | `python` | Pure Python fallback. |

The C++ training backend is semantically equivalent to the reference minBPE implementation (verified via cross-validation scripts).

## Rank-limit Encoding Modes

The engine supports limiting which merge rules are applied at encoding time, without modifying the codebook:

| Mode | Behavior |
|------|----------|
| `all` | Apply all rules (default) |
| `topk` | Only apply the top-k rules by rank |
| `random` | Sample k uniformly from [min, max] per batch |
| `gaussian` | Sample k from a truncated Gaussian biased toward the upper bound |

```python
# Fixed top-k
eng = BPEEngine(encode_backend='cpp', encode_rank_mode='topk', encode_rank_k=128)

# Random sampling per batch
eng = BPEEngine(encode_backend='cpp', encode_rank_mode='random',
                encode_rank_min=64, encode_rank_max=256)
```

Within a batch, k is sampled once and applied to all sequences (via C++ `batch_encode_with_limit`).

## Performance (qm9test, 2000 merges, min_freq=100)

- C++ training: ~0.33s (vs ~13s for reference minBPE)
- C++ encoding: >100K seq/s single-threaded

## Threading

`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set by default to avoid nested parallelism overhead. For DataLoader integration, construct a `BPEEngine` inside `worker_init_fn` to avoid pickling issues.
