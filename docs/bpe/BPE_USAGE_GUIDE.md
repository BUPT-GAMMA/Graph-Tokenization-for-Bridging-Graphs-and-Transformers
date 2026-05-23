# BPE Usage Guide

## How It Works

BPE compression is applied as a runtime transform rather than a static preprocessing step. You only need one serialized dataset — the BPE encoding happens on-the-fly during training. This makes it easy to experiment with different compression settings without re-processing data.

```
Graph → Serialization → Raw token sequences (cached on disk)
                              ↓
                         BPE Transform (at runtime, configurable)
                              ↓
                         Training / Inference
```

## Basic Usage

### Without BPE (raw sequences)

```bash
python run_pretrain.py --dataset qm9test --method feuler
python run_finetune.py --dataset qm9test --method feuler --task regression
```

### With BPE

```bash
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000
python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_num_merges 2000
```

**Important:** Pre-training and fine-tuning must use the same BPE settings (same merge count, same vocab).

## Parameters

### Core

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bpe_num_merges` | Number of BPE merges. 0 = no BPE. | 2000 |
| `--bpe_encode_backend` | Encoding backend | `cpp` |

### Encoding Strategy

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bpe_encode_rank_mode` | Which merge rules to apply | `all` |
| `--bpe_encode_rank_k` | K value for top-k mode | None |
| `--bpe_encode_rank_min` | Min range for random mode | None |
| `--bpe_encode_rank_max` | Max range for random mode | None |

### Evaluation Override

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bpe_eval_mode` | Override encoding mode at eval time | None |
| `--bpe_eval_topk` | Override top-k at eval time | None |

## Examples

### Deterministic compression (benchmarking)

```bash
python run_pretrain.py --dataset qm9test --method feuler \
    --bpe_num_merges 2000 --bpe_encode_rank_mode all
```

### Top-K compression (balance of compression and diversity)

```bash
python run_pretrain.py --dataset qm9test --method feuler \
    --bpe_num_merges 2000 --bpe_encode_rank_mode topk --bpe_encode_rank_k 1000
```

### Random compression (data augmentation)

```bash
# Random during training, deterministic during evaluation
python run_pretrain.py --dataset qm9test --method feuler \
    --bpe_num_merges 2000 \
    --bpe_encode_rank_mode random --bpe_encode_rank_min 100 --bpe_encode_rank_max 2000

python run_finetune.py --dataset qm9test --method feuler --task regression \
    --bpe_num_merges 2000 \
    --bpe_encode_rank_mode random --bpe_encode_rank_min 100 --bpe_encode_rank_max 2000 \
    --bpe_eval_mode all
```

## JSON Config

You can also pass BPE settings through a JSON config file:

```bash
python run_pretrain.py --dataset qm9test --method feuler --config_json bpe_config.json
```

```json
{
  "serialization": {
    "bpe": {
      "enabled": true,
      "encode_backend": "cpp",
      "encode_rank_mode": "topk",
      "encode_rank_k": 1000
    }
  }
}
```

## Best Practices

- **Pre-training**: random or gaussian mode can serve as data augmentation.
- **Fine-tuning**: use deterministic mode (`all`) for reproducibility.
- **Evaluation**: always deterministic.
- **C++ backend**: 5-10x faster than Python; use it whenever possible.
- **Consistency**: pre-training and fine-tuning must share the same BPE codebook and settings.

## Troubleshooting

**"BPE codebook not found"** — Run `prepare_data_new.py` first to train the BPE model.

**"Vocab size mismatch"** — The fine-tuning BPE settings don't match pre-training. Make sure `--bpe_num_merges` and the serialization method are identical.

**"encode_rank_k is only valid with topk mode"** — Check that your parameter combinations are consistent.
