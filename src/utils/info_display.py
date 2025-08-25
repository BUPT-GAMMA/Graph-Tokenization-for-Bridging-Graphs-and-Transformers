"""
信息展示工具 - 在关键阶段输出配置和模型信息
"""
from typing import Dict, Optional


def display_startup_config(logger,config, dataset_name: str, method: str, stage: str):
    """程序启动时显示关键配置信息"""
    logger.info("=" * 60)
    logger.info(f"🚀 {stage.upper()}阶段启动 - {dataset_name}/{method}")
    logger.info("=" * 60)
    
    # 核心配置
    encoder_type = config.encoder.type
    batch_size = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'batch_size', 'N/A')
    epochs = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'epochs', 'N/A')
    lr = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'learning_rate', 'N/A')
    
    if config.bert.pretraining.mlm_augmentation_methods is not None and len(config.bert.pretraining.mlm_augmentation_methods) > 0:
        use_augmentation = True
    elif config.bert.finetuning.regression_augmentation_methods is not None and len(config.bert.finetuning.regression_augmentation_methods) > 0:
        use_augmentation = True
    else:
        use_augmentation = False
        
    
    logger.info("📋 核心配置:")
    logger.info(f"   数据集={dataset_name}, 序列化={method}, 编码器={encoder_type}, 使用增强={use_augmentation}")
    logger.info(f"   训练参数: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    
    
    # 架构配置
    arch = config.bert.architecture
    model_desc = f"{arch.hidden_size}d_{arch.num_hidden_layers}l_{arch.num_attention_heads}h"
    logger.info(f"   模型架构: {model_desc}, 设备: {config.device}")
    logger.info(f"   最大序列长度: {arch.max_seq_length}, 最大位置嵌入长度: {arch.max_position_embeddings}")
    logger.info("=" * 60)


def display_data_info(logger, train_size: int, val_size: int, test_size: int, 
                     vocab_size: int, effective_max_length: Optional[int] = None):
    """数据加载完成后显示数据信息"""
    logger.info("📊 数据概览:")
    logger.info(f"   train/val/test={train_size:,} / {val_size:,} / {test_size:,} 样本, 词表大小={vocab_size:,} tokens")
    if effective_max_length:
        logger.info(f"   有效序列长度={effective_max_length}")


def display_model_info(logger, model, task_type: str, encoder_type: str):
    """模型创建完成后显示模型信息"""
    logger.info("🏗️ 模型概览:")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    task_head_params = total_params - encoder_params
    
    logger.info(f"   编码器={encoder_type} , 任务类型={task_type}, 输出维度={model.output_dim}")
    logger.info(f"   可学习参数/总参数= {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {total_params:,} (encoder/task_head={encoder_params:,} / {task_head_params:,}) ")


def display_training_setup(logger, total_steps: int, steps_per_epoch: int, epochs: int,
                          optimizer_info: str, scheduler_info: str = None):
    """训练开始前显示训练设置"""
    logger.info("⚡ 训练设置:")
    logger.info(f"   总步数: {total_steps:,} ({steps_per_epoch} steps/epoch × {epochs} epochs)")
    logger.info(f"   优化器: {optimizer_info}, 调度器: {scheduler_info}")


def display_stage_separator(logger, stage_name: str, description: str = ""):
    """阶段分隔符"""
    # separator = "=" * 50
    # logger.info(separator)
    if description:
        logger.info(f"===========🎯 {stage_name}: {description}===========")
    else:
        logger.info(f"===========🎯 {stage_name}===========")
    # logger.info(separator)

def display_performance_summary(logger, total_time: float, total_samples: int, 
                               best_metric: float, best_epoch: int,
                               stage: str = "训练"):
    """性能总结"""
    throughput = total_samples / total_time if total_time > 0 else 0
    logger.info(f"📈 {stage}性能总结:")
    logger.info(f"   用时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"   速度: {throughput:.1f} samples/s")
    logger.info(f"   最优结果: {best_metric:.4f} (epoch {best_epoch})")

