"""
信息展示工具 - 在关键阶段输出配置和模型信息
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def display_startup_config(config, dataset_name: str, method: str, stage: str):
    """程序启动时显示关键配置信息"""
    logger.info("=" * 60)
    logger.info(f"🚀 {stage.upper()}阶段启动 - {dataset_name}/{method}")
    logger.info("=" * 60)
    
    # 核心配置
    encoder_type = config.encoder.type
    batch_size = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'batch_size', 'N/A')
    epochs = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'epochs', 'N/A')
    lr = getattr(config.bert.pretraining if stage == "预训练" else config.bert.finetuning, 'learning_rate', 'N/A')
    
    logger.info("📋 核心配置:")
    logger.info(f"   数据集: {dataset_name}")
    logger.info(f"   序列化: {method}")
    logger.info(f"   编码器: {encoder_type}")
    logger.info(f"   训练参数: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    
    # 架构配置
    arch = config.bert.architecture
    model_desc = f"{arch.hidden_size}d_{arch.num_hidden_layers}l_{arch.num_attention_heads}h"
    logger.info(f"   模型架构: {model_desc}")
    logger.info(f"   最大序列长度: {arch.max_seq_length} (配置值)")
    if hasattr(arch, 'max_position_embeddings'):
        logger.info(f"   位置嵌入长度: {arch.max_position_embeddings}")
    
    # 设备和系统
    logger.info(f"   设备: {config.device}")
    logger.info(f"   工作线程: {config.system.num_workers}")
    logger.info("=" * 60)


def display_data_info(train_size: int, val_size: int, test_size: int, 
                     vocab_size: int, effective_max_length: Optional[int] = None):
    """数据加载完成后显示数据信息"""
    logger.info("📊 数据概览:")
    logger.info(f"   训练集: {train_size:,} 样本")
    logger.info(f"   验证集: {val_size:,} 样本") 
    logger.info(f"   测试集: {test_size:,} 样本")
    logger.info(f"   词表大小: {vocab_size:,} tokens")
    if effective_max_length:
        logger.info(f"   有效序列长度: {effective_max_length}")


def display_model_info(model, task_type: str, encoder_type: str, vocab_size: int):
    """模型创建完成后显示模型信息"""
    logger.info("🏗️ 模型概览:")
    
    # 编码器信息
    hidden_size = model.encoder.get_hidden_size()
    max_seq_length = model.encoder.get_max_seq_length()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    task_head_params = total_params - encoder_params
    
    logger.info(f"   编码器: {encoder_type} ({hidden_size}d, max_len={max_seq_length})")
    logger.info(f"   任务类型: {task_type}")
    logger.info(f"   输出维度: {model.output_dim}")
    logger.info(f"   参数统计: 总计{total_params:,} = 编码器{encoder_params:,} + 任务头{task_head_params:,}")


def display_training_setup(total_steps: int, steps_per_epoch: int, epochs: int,
                          optimizer_info: str, scheduler_info: str = None):
    """训练开始前显示训练设置"""
    logger.info("⚡ 训练设置:")
    logger.info(f"   总步数: {total_steps:,} ({steps_per_epoch} steps/epoch × {epochs} epochs)")
    logger.info(f"   优化器: {optimizer_info}")
    if scheduler_info:
        logger.info(f"   调度器: {scheduler_info}")


def display_stage_separator(stage_name: str, description: str = ""):
    """阶段分隔符"""
    separator = "=" * 50
    logger.info(separator)
    if description:
        logger.info(f"🎯 {stage_name}: {description}")
    else:
        logger.info(f"🎯 {stage_name}")
    logger.info(separator)


def display_config_mismatch_warning(pretrain_config: Dict, current_config: Dict):
    """显示配置不匹配警告"""
    mismatches = []
    
    # 检查关键配置项
    key_configs = [
        ('max_seq_length', '最大序列长度'),
        ('vocab_size', '词表大小'),
        ('hidden_size', '隐藏维度'),
        ('num_hidden_layers', '层数'),
        ('num_attention_heads', '注意力头数')
    ]
    
    for key, desc in key_configs:
        pretrain_val = pretrain_config.get(key)
        current_val = current_config.get(key)
        if pretrain_val and current_val and pretrain_val != current_val:
            mismatches.append(f"   {desc}: {pretrain_val} → {current_val}")
    
    if mismatches:
        logger.warning("⚠️ 配置不匹配警告:")
        for mismatch in mismatches:
            logger.warning(mismatch)
        logger.warning("💡 建议检查预训练和微调的配置一致性")
    else:
        logger.info("✅ 配置兼容性检查通过")


def display_performance_summary(total_time: float, total_samples: int, 
                               best_metric: float, best_epoch: int,
                               stage: str = "训练"):
    """性能总结"""
    throughput = total_samples / total_time if total_time > 0 else 0
    logger.info(f"📈 {stage}性能总结:")
    logger.info(f"   用时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"   速度: {throughput:.1f} samples/s")
    logger.info(f"   最优结果: {best_metric:.4f} (epoch {best_epoch})")

