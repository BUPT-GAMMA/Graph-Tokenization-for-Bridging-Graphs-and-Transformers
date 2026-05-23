# BERT Encoder

This directory contains the BERT-based encoder components used by the project. In normal usage, you don't interact with these files directly — they are accessed through the unified encoder interface (`src/models/unified_encoder.py`) and the training pipelines (`src/training/`).

## Components

| File | Purpose |
|------|---------|
| `vocab_manager.py` | Vocabulary management: build from token sequences, encode/decode, save/load |
| `model.py` | BERT MLM model and regression model definitions |
| `data.py` | Dataset classes for MLM and regression tasks |

## How It Fits Together

```
Token ID sequences → VocabManager (build vocab) → BertEncoder (from unified_encoder.py)
                                                       ↓
                                              MLM pre-training (pretrain_pipeline.py)
                                                       ↓
                                              Fine-tuning (finetune_pipeline.py)
```

## Encoder Creation

The recommended way to create a BERT encoder:

```python
from src.models.unified_encoder import create_encoder_from_config
encoder = create_encoder_from_config(model_name, encoder_config_dict, vocab_manager)
```

Or with explicit parameters:

```python
from src.models.unified_encoder import create_encoder
encoder = create_encoder("bert", vocab_size=5000, pad_token_id=0,
                         hidden_size=512, num_hidden_layers=4, ...)
```

## Vocabulary

```python
from src.models.bert.vocab_manager import VocabManager

# Build from token sequences
vocab_manager = VocabManager.from_config(config)

# Or construct with explicit parameters
vocab_manager = VocabManager(pad_token="[PAD]", mask_token="[MASK]", ...)
```

Special tokens are reserved at fixed IDs: PAD(0), UNK(1), CLS(2), SEP(3), MASK(4).

## Architecture

- Based on HuggingFace `BertModel` with configurable architecture (hidden size, layers, heads)
- MLM head for pre-training (80% mask / 10% random / 10% keep)
- Regression head: mean pooling over non-padding tokens → MLP → scalar output
- Pooling options: `mean` (default), `cls`, `max`