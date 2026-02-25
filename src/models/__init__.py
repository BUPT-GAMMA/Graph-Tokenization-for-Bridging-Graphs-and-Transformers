"""Models module — unified encoder interface.
模型模块 - 统一编码器接口

Features / 功能:
- Unified encoder factory / 统一编码器工厂
- Supports BERT, GTE, and other encoders / 支持 BERT、GTE 等多种编码器
- Unified task model creation / 统一任务模型创建
- Extensible model registration / 可扩展的模型注册

Example::

    >>> from src.models import create_encoder, create_task_model, list_supported_encoders
    >>> print(list_supported_encoders())
    >>> bert_encoder = create_encoder('bert', {'hidden_size': 512, 'num_hidden_layers': 4}, vocab_manager)
    >>> gte_encoder = create_encoder('Alibaba-NLP/gte-multilingual-base', gte_config, vocab_manager)
    >>> task_model = create_task_model('Alibaba-NLP/gte-multilingual-base', gte_config, vocab_manager,
    ...                               'regression', output_dim=1, pooling_method='mean')
"""

from .unified_encoder import (
    # Core API
    create_encoder,
    
    # Base classes
    BaseEncoder,
    BertEncoder,
    GTEEncoder,
    
    # Utilities
    list_supported_encoders,
)

# Public interface
__all__ = [
    'create_encoder',
    'list_supported_encoders',
    'BaseEncoder',
    'BertEncoder', 
    'GTEEncoder',
]


# Preset configurations
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
    """Get preset model config."""
    if model_name not in PRESET_CONFIGS:
        raise ValueError(f"No preset config for model: {model_name}")
    return PRESET_CONFIGS[model_name].copy()


def create_encoder_with_preset(
    model_name: str,
    vocab_manager,
    config_overrides: dict = None
):
    """Create encoder with preset config."""
    config = get_preset_config(model_name)
    if config_overrides:
        config.update(config_overrides)
    from .unified_encoder import create_encoder_from_config
    return create_encoder_from_config(model_name, config, vocab_manager)
