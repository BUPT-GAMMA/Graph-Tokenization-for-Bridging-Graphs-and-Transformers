"""
统一模型创建接口
================

基于BaseEncoder的统一模型创建，支持所有编码器类型。
任务相关逻辑由TaskHandler处理。
"""

from __future__ import annotations

from src.models.unified_encoder import BaseEncoder
from src.models.universal_model import UniversalModel
from src.models.model_factory import create_universal_model


# create_unified_model已废弃，现在使用src/models/model_factory.py中的create_universal_model


def create_model_from_udi(udi, pretrained_model=None, pooling_method: str = 'mean'):
    """
    统一模型创建接口 - 支持自动预训练加载（上层接口保持兼容）
    
    Args:
        udi: UnifiedDataInterface实例
        pretrained_model: 预训练模型（可选）
                         - 如果为None，则从默认路径自动加载预训练模型
                         - 如果提供，则使用该预训练模型的权重
        pooling_method: 池化方法
        
    Returns:
        (model, task_handler) 元组 - 与原接口完全兼容
    """
    
    # 1. 如果没有提供预训练模型，则自动加载
    if pretrained_model is None:
        from src.training.model_builder import load_pretrained_backbone
        try:
            pretrained_model = load_pretrained_backbone(udi.config)
            print("🔄 自动加载预训练模型成功")
        except Exception as e:
            print(f"⚠️  预训练模型加载失败，将使用随机初始化: {e}")
            pretrained_model = None
    
    # 2. 从UDI推断任务类型
    task_type = udi.get_dataset_task_type()
    
    # 3. 获取词表管理器
    if pretrained_model is not None and hasattr(pretrained_model, 'vocab_manager'):
        vocab_manager = pretrained_model.vocab_manager
    else:
        # 🔧 修复：get_vocab()需要method参数
        method = udi.config.serialization.method
        vocab_manager = udi.get_vocab(method=method)
    
    # 4. 调用统一创建接口
    model, task_handler = create_universal_model(
        config=udi.config,
        vocab_manager=vocab_manager,
        task_type=task_type,
        udi=udi
    )
    
    # 5. 复制预训练权重（如果存在）
    if pretrained_model is not None:
        _copy_pretrained_weights(model, pretrained_model)
        print("✅ 预训练权重复制完成")
    
    return model, task_handler


def _copy_pretrained_weights(target_model: UniversalModel, source_model):
    """
    权重复制 - 支持多种源模型类型
    
    Args:
        target_model: 目标UniversalModel
        source_model: 源模型（BertMLM或BaseEncoder）
    """
    
    if hasattr(source_model, 'bert'):
        # BertMLM类型：复制BERT编码器权重
        # source_model.bert → target_model.encoder.bert (新的直接结构)
        if hasattr(target_model.encoder, 'bert'):
            try:
                target_model.encoder.bert.load_state_dict(source_model.bert.state_dict())
                print("🔄 BertMLM权重 → UniversalModel.encoder.bert 复制成功")
            except Exception as e:
                print(f"⚠️  BERT权重复制失败: {e}")
                raise
        else:
            raise ValueError("目标模型的编码器结构与BERT不兼容")
            
    elif isinstance(source_model, BaseEncoder):
        # BaseEncoder类型：直接复制编码器权重
        try:
            target_model.encoder.load_state_dict(source_model.state_dict())
            print("🔄 BaseEncoder权重 → UniversalModel.encoder 复制成功")
        except Exception as e:
            print(f"⚠️  编码器权重复制失败: {e}")
            raise
            
    elif hasattr(source_model, 'encoder'):
        # UniversalModel类型：复制编码器权重
        try:
            target_model.encoder.load_state_dict(source_model.encoder.state_dict())
            print("🔄 UniversalModel.encoder权重 → UniversalModel.encoder 复制成功")
        except Exception as e:
            print(f"⚠️  UniversalModel编码器权重复制失败: {e}")
            raise
    else:
        raise ValueError(f"不支持的源模型类型: {type(source_model)}")


# 废弃接口已清理