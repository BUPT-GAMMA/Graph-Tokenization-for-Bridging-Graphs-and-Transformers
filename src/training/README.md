# Training

This module implements the full training lifecycle: model construction, pre-training (MLM), fine-tuning (downstream tasks), evaluation, and optimization utilities.

## Workflow

```
prepare_data_new.py          run_pretrain.py              run_finetune.py
       ↓                           ↓                            ↓
  Serialization + BPE      pretrain_pipeline.py          finetune_pipeline.py
                                   ↓                            ↓
                            model_builder.py ←──────── model_builder.py
                                   ↓                            ↓
                             loops.train_epoch            loops.train_epoch
                                   ↓                     evaluate.evaluate_model
                             Save checkpoint                    ↓
                                                         Save best model
```

## Components

| File | Purpose |
|------|---------|
| `pretrain_pipeline.py` | `train_bert_mlm()` — full MLM pre-training loop with logging, checkpointing, and optional Optuna pruning |
| `finetune_pipeline.py` | `run_finetune()` — fine-tuning on downstream tasks (regression, classification, multi-label) |
| `model_builder.py` | `build_task_model()` — unified model construction entry point; handles encoder creation, pre-trained weight loading, and task head setup |
| `task_handler.py` | `TaskHandler` — manages loss functions, output post-processing, and metric computation per task type |
| `tasks.py` | `build_regression_loaders()`, `build_classification_loaders()` — dataset/dataloader construction for fine-tuning |
| `loops.py` | `train_epoch()`, `evaluate_epoch()` — generic training and evaluation loops |
| `evaluate.py` | `evaluate_model()` — comprehensive evaluation with per-task metrics and optional multi-variant aggregation |
| `optim.py` | `build_from_config()` — optimizer and scheduler construction from config |
| `augmentation.py` | BPE-based data augmentation utilities |
| `learned_aggregation.py` | Learned aggregation for multi-variant fine-tuning (variant weighting) |

## Pre-training

```bash
python run_pretrain.py \
    --dataset qm9test --method feuler \
    --experiment_group my_exp \
    --epochs 100 --batch_size 256
```

The pipeline:
1. Loads serialized sequences and vocab via `UnifiedDataInterface`
2. Builds `UniversalModel` with MLM task head
3. Runs training loop with TensorBoard logging
4. Saves checkpoints to `model/{group}/{dataset}-{method}/`

## Fine-tuning

```bash
python run_finetune.py \
    --dataset qm9test --method feuler \
    --experiment_group my_exp \
    --target_property homo \
    --epochs 200 --batch_size 64
```

The pipeline:
1. Loads data and builds task-specific dataloaders
2. Builds model and loads pre-trained encoder weights
3. Trains with early stopping based on validation metrics
4. Reports test metrics at best validation checkpoint

## Evaluation Metrics

| Task type | Metrics |
|-----------|---------|
| Regression | MAE, RMSE, R², Pearson r |
| Classification | Accuracy, F1, ROC-AUC |
| Multi-label classification | AP (average precision) |
| Multi-target regression | Per-target MAE |

## Multi-variant Aggregation

When using multiple serialization samples per graph, fine-tuning supports three aggregation modes:

- **`avg`** — average predictions across variants
- **`best`** — select the best single variant
- **`learned`** — train a `VariantWeightingAggregator` to weight variants (default)
