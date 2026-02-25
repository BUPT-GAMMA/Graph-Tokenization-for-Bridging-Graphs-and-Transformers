"""
Info display utilities - print config and model info at key stages.
信息展示工具 - 在关键阶段打印配置和模型信息。
"""
from typing import Dict, Optional


def display_startup_config(logger,config, dataset_name: str, method: str, stage: str):
    """Display key config info at startup."""
    logger.info("=" * 60)
    logger.info(f"🚀 {stage.upper()} starting - {dataset_name}/{method}")
    logger.info("=" * 60)
    
    # Core config
    encoder_type = config.encoder.type
    batch_size = getattr(config.bert.pretraining if stage == "pretrain" else config.bert.finetuning, 'batch_size', 'N/A')
    epochs = getattr(config.bert.pretraining if stage == "pretrain" else config.bert.finetuning, 'epochs', 'N/A')
    lr = getattr(config.bert.pretraining if stage == "pretrain" else config.bert.finetuning, 'learning_rate', 'N/A')
    
    if config.bert.pretraining.mlm_augmentation_methods is not None and len(config.bert.pretraining.mlm_augmentation_methods) > 0:
        use_augmentation = True
    elif config.bert.finetuning.regression_augmentation_methods is not None and len(config.bert.finetuning.regression_augmentation_methods) > 0:
        use_augmentation = True
    else:
        use_augmentation = False
        
    
    logger.info("📋 Config:")
    logger.info(f"   dataset={dataset_name}, method={method}, encoder={encoder_type}, augmentation={use_augmentation}")
    logger.info(f"   Training: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    
    
    # Architecture config
    arch = config.bert.architecture
    model_desc = f"{arch.hidden_size}d_{arch.num_hidden_layers}l_{arch.num_attention_heads}h"
    logger.info(f"   Model: {model_desc}, device: {config.device}")
    logger.info(f"   max_seq_length: {arch.max_seq_length}, max_position_embeddings: {arch.max_position_embeddings}")
    logger.info("=" * 60)


def display_data_info(logger, train_size: int, val_size: int, test_size: int, 
                     vocab_size: int, effective_max_length: Optional[int] = None):
    """Display data info after loading."""
    logger.info("📊 Data overview:")
    logger.info(f"   train/val/test={train_size:,} / {val_size:,} / {test_size:,} samples, vocab_size={vocab_size:,} tokens")
    if effective_max_length:
        logger.info(f"   effective_max_length={effective_max_length}")


def display_model_info(logger, model, task_type: str, encoder_type: str):
    """Display model info after creation."""
    logger.info("🏗️ Model overview:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    task_head_params = total_params - encoder_params
    
    logger.info(f"   encoder={encoder_type}, task={task_type}, output_dim={model.output_dim}")
    logger.info(f"   trainable/total= {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {total_params:,} (encoder/task_head={encoder_params:,} / {task_head_params:,}) ")


def display_training_setup(logger, total_steps: int, steps_per_epoch: int, epochs: int,
                          optimizer_info: str, scheduler_info: str = None):
    """Display training setup before start."""
    logger.info("⚡ Training setup:")
    logger.info(f"   Total steps: {total_steps:,} ({steps_per_epoch} steps/epoch x {epochs} epochs)")
    logger.info(f"   Optimizer: {optimizer_info}, Scheduler: {scheduler_info}")


def display_stage_separator(logger, stage_name: str, description: str = ""):
    """Stage separator."""
    # separator = "=" * 50
    # logger.info(separator)
    if description:
        logger.info(f"===========🎯 {stage_name}: {description}===========")
    else:
        logger.info(f"===========🎯 {stage_name}===========")
    # logger.info(separator)

def display_performance_summary(logger, total_time: float, total_samples: int, 
                               best_metric: float, best_epoch: int,
                               stage: str = "training"):
    """Performance summary."""
    throughput = total_samples / total_time if total_time > 0 else 0
    logger.info(f"📈 {stage} summary:")
    logger.info(f"   Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"   Throughput: {throughput:.1f} samples/s")
    logger.info(f"   Best: {best_metric:.4f} (epoch {best_epoch})")

