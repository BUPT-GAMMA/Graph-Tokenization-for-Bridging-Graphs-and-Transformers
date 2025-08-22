"""
Models Module
=============

统一的模型接口模块，提供多种编码器的统一访问。

主要功能：
- 统一编码器工厂接口
- 支持BERT、GTE等多种编码器
- 统一的任务模型创建
- 可扩展的模型注册机制

使用示例：
    >>> from src.models import create_encoder, create_task_model, list_supported_encoders
    >>> 
    >>> # 查看支持的编码器
    >>> print(list_supported_encoders())
    >>> 
    >>> # 创建BERT编码器
    >>> bert_config = {'hidden_size': 512, 'num_hidden_layers': 4}
    >>> bert_encoder = create_encoder('bert', bert_config, vocab_manager)
    >>> 
    >>> # 创建GTE编码器  
    >>> gte_config = {'hidden_size': 768, 'optimization': {'unpad_inputs': True}}
    >>> gte_encoder = create_encoder('Alibaba-NLP/gte-multilingual-base', gte_config, vocab_manager)
    >>> 
    >>> # 创建任务模型
    >>> task_model = create_task_model('Alibaba-NLP/gte-multilingual-base', gte_config, vocab_manager, 
    ...                               'regression', output_dim=1, pooling_method='mean')
"""

from .unified_encoder import (
    # 核心接口
    create_encoder,
    
    # 基础类
    BaseEncoder,
    BertEncoder,
    GTEEncoder,
    
    # 工具函数
    list_supported_encoders,
)

# 导出的公共接口
__all__ = [
    'create_encoder',
    'list_supported_encoders',
    'BaseEncoder',
    'BertEncoder', 
    'GTEEncoder',
]


# 便捷的预设配置
PRESET_CONFIGS = {
    'bert-small': {
        'hidden_size': 512,
        'num_hidden_layers': 4,
        'num_attention_heads': 8,
        'intermediate_size': 2048,
        'max_position_embeddings': 512,
        'dropout': 0.1,
    },
    
    'Alibaba-NLP/gte-multilingual-base': {
        'hidden_size': 768,
        'max_seq_length': 8192,
        'optimization': {
            'unpad_inputs': True,
            'use_memory_efficient_attention': True,
            'torch_dtype': 'float16'
        }
    }
}


def get_preset_config(model_name: str) -> dict:
    """获取预设的模型配置"""
    if model_name not in PRESET_CONFIGS:
        raise ValueError(f"没有预设配置的模型: {model_name}")
    return PRESET_CONFIGS[model_name].copy()


def create_encoder_with_preset(
    model_name: str,
    vocab_manager,
    config_overrides: dict = None
):
    """使用预设配置创建编码器"""
    config = get_preset_config(model_name)
    if config_overrides:
        config.update(config_overrides)
    return create_encoder(model_name, config, vocab_manager)
