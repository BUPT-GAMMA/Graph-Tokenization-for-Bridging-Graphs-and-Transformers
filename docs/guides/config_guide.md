# Configuration Guide

## How Configuration Works

All configuration is managed through `config.py` (a `ProjectConfig` class) and `config/default_config.yml`. The system follows a layered override model:

```
default_config.yml  →  config.py defaults  →  command-line arguments  →  JSON config file
```

Command-line arguments take the highest priority. Each experiment run saves a full config snapshot to its log directory for reproducibility.

## Key Config Sections

### System

```yaml
system:
  device: auto       # "auto", "cuda", "cpu"
  seed: 42
```

### Dataset

```yaml
dataset:
  name: qm9
  limit: null         # set an integer to limit dataset size for debugging
```

### Serialization & BPE

```yaml
serialization:
  method: feuler      # feuler, eulerian, cpp, dfs, bfs, ...
  bpe:
    num_merges: 2000
    min_frequency: 100
    encode_backend: cpp
    encode_rank_mode: all   # all, topk, random, gaussian
```

### Model (BERT)

```yaml
bert:
  architecture:
    hidden_size: 512
    num_hidden_layers: 4
    num_attention_heads: 8
    intermediate_size: 2048
    max_seq_length: 64
    hidden_dropout_prob: 0.1
```

For GTE, the architecture is fixed (768-dim, 12 layers) and loaded from the pretrained config. Only `reset_weights` and optimization flags are configurable.

### Training

```yaml
training:
  pretrain:
    epochs: 100
    batch_size: 256
    learning_rate: 1e-4
  finetune:
    epochs: 200
    batch_size: 64
    learning_rate: 2e-5
```

### Experiment

```yaml
experiment:
  experiment_group: my_group   # groups related runs together
  repeat_runs: 3               # number of repeated runs for statistics
```

## Command-Line Override

Most config values can be overridden from the command line. For example:

```bash
python run_pretrain.py \
    --dataset qm9 \
    --method feuler \
    --experiment_group ablation_v1 \
    --epochs 50 \
    --batch_size 128 \
    --learning_rate 2e-4
```

You can also pass a JSON file for more complex overrides:

```bash
python run_pretrain.py --config_json my_config.json
```

## Path Layout

The config system manages these directories automatically:

| Path | Purpose |
|------|---------|
| `data/` | Raw dataset files |
| `data/processed/` | Serialized sequences, vocabularies |
| `model/bpe/` | Trained BPE codebooks |
| `model/{group}/` | Saved model checkpoints |
| `log/{group}/` | Training logs and metrics |

## Environment Variables

Only system-level settings use environment variables:

- `CUDA_VISIBLE_DEVICES` — GPU selection
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS` — CPU thread control (set automatically in `config.py`)

Experiment parameters should always go through the config system, not environment variables.

## Tips

- Always set `--experiment_group` to keep runs organized.
- Use `--debug` flag for quick smoke tests (automatically reduces dataset size and epochs).
- Config snapshots are saved as JSON in the log directory — check them if results look unexpected.
- When fine-tuning, the BPE settings must match pre-training exactly.
