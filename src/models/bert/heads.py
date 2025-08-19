"""
统一模型创建接口
================

创建统一的BERT模型，任务相关逻辑由TaskHandler处理。
"""

from __future__ import annotations
from typing import Optional

from src.models.bert.unified_model import BertUnified
from src.models.bert import BertConfig
from src.training.task_handler import create_task_handler


def create_unified_model(
    *,
    vocab_manager,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    pooling_method: str,
    dropout: float,
    max_position_embeddings: int,
    layer_norm_eps: float,
    output_dim: int,
):
    """
    创建统一的BERT模型
    
    这个函数创建一个真正的统一模型，不区分任务类型。
    任务相关的逻辑（损失函数、指标等）由TaskHandler处理。
    """
    cfg = BertConfig(
        vocab_size=vocab_manager.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        dropout=dropout,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        pad_token_id=vocab_manager.pad_token_id,
    )
    
    return BertUnified(
        config=cfg,
        vocab_manager=vocab_manager,
        output_dim=output_dim,
        pooling_method=pooling_method,
    )


def create_model_from_udi(udi, pretrained_model, pooling_method: str = 'mean'):
    """
    根据UDI创建模型（推荐方式）
    
    Args:
        udi: UnifiedDataInterface实例
        pretrained_model: 预训练的BERT模型
        pooling_method: 池化方法
        
    Returns:
        (model, task_handler) 元组
    """
    # 从UDI获取任务信息
    task_handler = create_task_handler(udi)
    
    # 创建统一模型
    model = create_unified_model(
        vocab_manager=pretrained_model.vocab_manager,
        hidden_size=pretrained_model.config.hidden_size,
        num_hidden_layers=pretrained_model.config.num_hidden_layers,
        num_attention_heads=pretrained_model.config.num_attention_heads,
        intermediate_size=pretrained_model.config.intermediate_size,
        pooling_method=pooling_method,
        dropout=pretrained_model.config.dropout,
        max_position_embeddings=pretrained_model.config.max_position_embeddings,
        layer_norm_eps=pretrained_model.config.layer_norm_eps,
        output_dim=task_handler.output_dim,
    )
    
    # 复制预训练权重到新模型
    if hasattr(pretrained_model, 'bert'):
        model.bert.load_state_dict(pretrained_model.bert.state_dict())
    
    return model, task_handler


# 兼容旧接口（将被废弃）
def create_task_head(*args, **kwargs):
    """
    废弃警告：请使用 create_model_from_udi
    """
    import warnings
    warnings.warn(
        "create_task_head 已废弃，请使用 create_model_from_udi",
        DeprecationWarning,
        stacklevel=2
    )
    # 这里可以添加向后兼容的代码
    raise NotImplementedError("请使用新的 create_model_from_udi 接口")