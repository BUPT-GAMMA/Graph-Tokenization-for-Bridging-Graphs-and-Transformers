"""
统一模型工厂
============

create_universal_model - 统一的模型创建接口
支持预训练(MLM)和微调(分类/回归)的无缝切换
"""

from __future__ import annotations
from typing import Tuple, Optional, Dict
import torch

from src.models.universal_model import UniversalModel
from src.models.unified_encoder import create_encoder
from src.training.task_handler import TaskHandler, create_task_handler


def create_universal_model(
    config,  # ProjectConfig
    vocab_manager,  # VocabManager
    task_type: str,           # 'mlm', 'classification', 'regression'
    output_dim: int = None,   # MLM时自动设为vocab_size
    udi = None  # UnifiedDataInterface, 可选
) -> Tuple[UniversalModel, TaskHandler]:
    """
    统一模型创建 - 支持所有任务类型
    
    Args:
        config: 项目配置
        vocab_manager: 词表管理器
        task_type: 任务类型 ('mlm', 'classification', 'regression', 等)
        output_dim: 输出维度，MLM时可留空自动设置
        udi: 统一数据接口，用于推断输出维度
        
    Returns:
        (UniversalModel, TaskHandler) 元组
    """
    
    # 1. 创建编码器
    encoder_type = config.encoder.type
    encoder_config = _build_encoder_config(config, encoder_type, task_type)  # 🆕 传递task_type
    encoder = create_encoder(encoder_type, encoder_config, vocab_manager)
    print(f"🔧 创建编码器: {encoder_type} ({encoder.get_hidden_size()}维)")
    
    # 2. 确定输出维度
    if task_type == 'mlm':
        output_dim = vocab_manager.vocab_size
        print(f"🔤 MLM任务输出维度: {output_dim} (词表大小)")
    elif output_dim is None and udi:
        output_dim = _get_output_dim_from_udi(udi, task_type)
        print(f"🎯 {task_type}任务输出维度: {output_dim} (从UDI推断)")
    elif output_dim is None:
        raise ValueError(f"任务类型 {task_type} 需要指定 output_dim 或提供 udi")
    
    # 3. 创建任务处理器
    if task_type == 'mlm':
        task_handler = create_task_handler(task_type='mlm', vocab_size=vocab_manager.vocab_size)
    else:
        task_handler = create_task_handler(udi=udi)
    print(f"📋 任务处理器: {task_type} (损失函数: {type(task_handler.loss_fn).__name__})")
    
    # 4. 获取任务头配置
    task_head_config = _get_task_head_config(config)
    
    # 5. 创建统一模型
    model = UniversalModel(
        encoder=encoder,
        task_type=task_type,
        output_dim=output_dim,
        pooling_method=config.bert.architecture.pooling_method,
        task_head_config=task_head_config
    )
    
    print(f"🎯 UniversalModel创建成功: {task_type} 任务")
    return model, task_handler


def _build_encoder_config(config, encoder_type: str, task_type: str = None) -> Dict:
    """构建编码器配置"""
    
    if encoder_type == 'bert':
        # BERT编码器配置
        return {
            'hidden_size': config.bert.architecture.hidden_size,
            'num_hidden_layers': config.bert.architecture.num_hidden_layers,
            'num_attention_heads': config.bert.architecture.num_attention_heads,
            'intermediate_size': config.bert.architecture.intermediate_size,
            'hidden_dropout_prob': config.bert.architecture.hidden_dropout_prob,
            'attention_probs_dropout_prob': config.bert.architecture.attention_probs_dropout_prob,
            'max_position_embeddings': config.bert.architecture.max_position_embeddings,
            'layer_norm_eps': config.bert.architecture.layer_norm_eps,
            'type_vocab_size': getattr(config.bert.architecture, 'type_vocab_size', 2),
            'initializer_range': getattr(config.bert.architecture, 'initializer_range', 0.02)
        }
    elif 'gte' in encoder_type.lower():
        # GTE编码器配置
        gte_config = {
            'hidden_size': 768,  # GTE固定768维
            'max_seq_length': 256,  # GTE支持长序列
            'optimization': {
                'unpad_inputs': True,
                'use_memory_efficient_attention': True,
                'torch_dtype': 'float32'  # 微调时使用float32
            }
        }
        
        # 🆕 根据配置决定是否重新初始化权重
        if getattr(config, '_reinit_weights', False):
            gte_config['reinit_weights'] = True
            print(f"🔄 --reinit_weights检测到，将重新初始化GTE整个模型权重 (任务: {task_type})")
        else:
            if task_type == 'mlm':
                print("🔄 GTE MLM预训练：保持原权重，适配新词表")
            else:
                print("🔄 GTE微调：保持原权重")
        
        return gte_config
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")


def _get_output_dim_from_udi(udi, task_type: str) -> int:
    """从UDI推断输出维度"""
    
    dataset_task_type = udi.get_dataset_task_type()
    
    if dataset_task_type in ['classification', 'binary_classification', 'multi_label_classification']:
        return udi.get_num_classes()
    elif dataset_task_type in ['regression']:
        return 1  # 单目标回归
    elif dataset_task_type in ['multi_target_regression']:
        return udi.get_num_classes()  # 多目标回归，复用num_classes
    else:
        raise ValueError(f"无法从UDI推断任务 {task_type} 的输出维度")


def _get_task_head_config(config) -> Dict:
    """获取任务头配置"""
    
    try:
        # 尝试从配置中获取任务头设置
        if hasattr(config.bert.architecture, 'task_head'):
            return {
                'hidden_ratio': config.bert.architecture.task_head.hidden_ratio,
                'activation': config.bert.architecture.task_head.activation,
                'dropout': config.bert.architecture.task_head.dropout
            }
    except AttributeError:
        pass
    
    # 返回默认配置
    return {
        'hidden_ratio': 0.5,
        'activation': 'relu',
        'dropout': 0.1
    }
