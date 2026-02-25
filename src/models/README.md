# Models

This module provides all model components: encoders, task heads, and the unified model that combines them.

## Architecture

```
Token IDs → Encoder (BERT or GTE) → sequence_output [B, T, H]
                                          ↓
                                    Pooling (mean / cls)
                                          ↓
                                    pooled [B, H]
                                          ↓
                                    TaskHead (MLP)
                                          ↓
                                    predictions [B, output_dim]
```

For MLM pre-training, the task head operates on the full sequence output (no pooling).

## Components

| File | Purpose |
|------|---------|
| `unified_encoder.py` | `BaseEncoder` ABC + `BertEncoder` / `GTEEncoder` implementations + factory `create_encoder` |
| `universal_model.py` | `UniversalModel` — wraps encoder + task head into a single `nn.Module` |
| `unified_task_head.py` | `UnifiedTaskHead` — builds MLP or linear head based on task type |
| `model_factory.py` | `create_universal_model` — top-level factory used by training pipelines |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `bert/` | BERT-specific components: config, vocab, dataset classes ([details](bert/README.md)) |
| `gte/` | GTE encoder integration (Alibaba-NLP/gte-multilingual-base) |
| `aggregators/` | `VariantWeightingAggregator` for multi-variant sequence weighting |
| `utils/` | Shared utilities (e.g., `pool_sequence`) |

## Supported Encoders

| Name | Class | Notes |
|------|-------|-------|
| `bert` | `BertEncoder` | HuggingFace `BertModel` with configurable architecture |
| `Alibaba-NLP/gte-multilingual-base` | `GTEEncoder` | Pre-trained GTE model, optionally with weight reset |

## Usage

Most users should go through `model_factory.create_universal_model()`, which is called by the training pipelines automatically. For direct use:

```python
from src.models.unified_encoder import create_encoder

encoder = create_encoder(
    "bert",
    vocab_size=5000,
    pad_token_id=0,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
)

# Get sentence-level representation
pooled = encoder.encode(input_ids, attention_mask, pooling_method="mean")  # [B, H]

# Get token-level representation (for MLM)
seq_out = encoder.get_sequence_output(input_ids, attention_mask)  # [B, T, H]
```

## Task Types

| Task | `task_type` | Output shape | Head |
|------|-------------|-------------|------|
| Masked LM | `mlm` | `[B, T, vocab_size]` | Linear |
| Classification | `classification` | `[B, num_classes]` | MLP |
| Binary classification | `binary_classification` | `[B, 1]` | MLP |
| Regression | `regression` | `[B, 1]` | MLP |
| Multi-target regression | `multi_target_regression` | `[B, num_targets]` | MLP |
| Multi-label classification | `multi_label_classification` | `[B, num_labels]` | MLP |
