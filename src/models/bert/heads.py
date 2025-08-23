"""
统一模型创建接口
================

基于BaseEncoder的统一模型创建，支持所有编码器类型。
任务相关逻辑由TaskHandler处理。
"""

from __future__ import annotations
from pathlib import Path
import torch
from typing import Dict

# 模型相关导入
from src.models.universal_model import UniversalModel
from src.models.unified_encoder import create_encoder
from src.training.task_handler import create_task_handler
from src.utils.logger import get_logger

# 创建模块级logger
logger = get_logger(__name__)


def create_model_from_udi(udi, pretrained_path: str = None, force_task_type: str = None):
    """
    统一模型创建接口 - 支持预训练和微调的统一入口
    
    流程设计：
    1. 自动判断任务类型（除非强制指定）
    2. 先完整创建模型（encoder + 任务头），此时决定是否reset权重
    3. 如果需要加载预训练，覆盖encoder权重
    4. 如果不需要加载预训练，直接返回（使用创建时的权重）
    
    Args:
        udi: UnifiedDataInterface实例  
        pretrained_path: 预训练模型路径（可选，预训练时应为None）
        force_task_type: 强制指定任务类型（如'mlm'用于预训练）
        
    Returns:
        (model, task_handler) 元组
    """
    logger.info("🏗️ 开始创建模型...")
    
    # === 1. 准备基础信息 ===
    method = udi.config.serialization.method
    pooling_method = udi.config.bert.architecture.pooling_method
    encoder_type = udi.config.encoder.type
    vocab_manager = udi.get_vocab(method=method)
    
    # 任务类型判断
    task_type = force_task_type if force_task_type is not None else udi.get_dataset_task_type()
    logger.info(f"🔧 配置: {task_type}任务, {encoder_type}编码器, {method}序列化")
    
    # === 第1阶段：创建完整模型 ===
    encoder = _create_complete_model(
        udi, task_type, encoder_type, vocab_manager, pooling_method
    )
    
    # 输出维度和任务处理器创建
    if task_type == 'mlm':
        output_dim = vocab_manager.vocab_size
        task_handler = create_task_handler(task_type='mlm', vocab_size=vocab_manager.vocab_size)
    elif task_type == 'regression':
        output_dim = 1
        task_handler = create_task_handler(udi=udi, task_type=task_type)
    elif task_type == 'binary_classification':
        output_dim = 2
        task_handler = create_task_handler(udi=udi, task_type=task_type)
    elif task_type in ['classification', 'multi_label_classification', 'multi_target_regression']:
        output_dim = udi.get_num_classes()
        task_handler = create_task_handler(udi=udi, task_type=task_type)
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")
    
    logger.info(f"🎯 任务: {task_type}, 输出维度: {output_dim}")
    
    # === 2.6. 创建完整模型 ===
    model = UniversalModel(
        encoder=encoder,
        task_type=task_type,
        output_dim=output_dim,
        pooling_method=pooling_method,
        task_head_config={'hidden_ratio': 0.5, 'activation': 'relu', 'dropout': 0.1}
    )
    # 第2阶段：权重处理
    if pretrained_path is not None:
        logger.info("🔄 加载预训练权重...")
        _load_and_copy_pretrained_weights(model, pretrained_path)
        logger.info("✅ 模型创建完成 (预训练权重)")
    else:
        weight_state = "重置权重" if udi.config.reset_weights else "默认初始化"
        logger.info(f"✅ 模型创建完成 ({weight_state})")
    return model, task_handler


def _build_encoder_config(config, encoder_type: str, task_type: str = None) -> Dict:
    """构建编码器配置 - 包含详细日志"""
    
    logger.info(f"🔧 构建编码器配置: {encoder_type}")
    logger.info(f"  任务类型: {task_type or 'unspecified'}")
    
    if encoder_type == 'bert':
        # BERT编码器配置
        bert_config = {
            'hidden_size': config.bert.architecture.hidden_size,
            'num_hidden_layers': config.bert.architecture.num_hidden_layers,
            'num_attention_heads': config.bert.architecture.num_attention_heads,
            'intermediate_size': config.bert.architecture.intermediate_size,
            'hidden_dropout_prob': config.bert.architecture.hidden_dropout_prob,
            'attention_probs_dropout_prob': config.bert.architecture.attention_probs_dropout_prob,
            'max_position_embeddings': config.bert.architecture.max_position_embeddings,
            'layer_norm_eps': config.bert.architecture.layer_norm_eps,
            'type_vocab_size': getattr(config.bert.architecture, 'type_vocab_size', 2),
            'initializer_range': getattr(config.bert.architecture, 'initializer_range', 0.02),
            # 统一传递 reset_weights
            'reset_weights': bool(config.reset_weights),
        }
        model_desc = f"{bert_config['hidden_size']}d_{bert_config['num_hidden_layers']}l_{bert_config['num_attention_heads']}h"
        logger.info(f"🔧 BERT配置: {model_desc}, max_len={bert_config['max_position_embeddings']}")
        return bert_config
        
    elif 'gte' in encoder_type.lower():
        # GTE编码器配置
        gte_config = {
            'hidden_size': 768,  # GTE固定768维
            'max_seq_length': 8096,  # GTE支持长序列
            'optimization': {
                'unpad_inputs': True,
                'use_memory_efficient_attention': True,
                'torch_dtype': 'float16'  
            }
        }
        
        # 统一 reset 字段（不再兼容reinit_weights）
        reset_weights = bool(config.reset_weights)
        if reset_weights:
            gte_config['reset_weights'] = True
            logger.info(f"🔄 检测到 reset_weights，将重新初始化GTE整个模型权重,任务类型: {task_type}")
            logger.warning("  注意：这会丢弃GTE的预训练权重！")
        else:
            logger.info("📋 GTE权重策略:")
            if task_type == 'mlm':
                logger.info("  - MLM预训练：保持GTE原始权重，仅适配新词表")
            else:
                logger.info("  - 微调任务：保持GTE原始权重")
        
        reinit_status = "重置权重" if gte_config.get('reset_weights', False) else "保持原权重"
        logger.info(f"🔧 GTE配置: {gte_config['hidden_size']}d, max_len={gte_config['max_seq_length']}, {reinit_status}")
        
        return gte_config
    else:
        logger.error(f"❌ 不支持的编码器类型: {encoder_type}")
        logger.info("📋 支持的编码器类型: bert, Alibaba-NLP/gte-multilingual-base")
        raise ValueError(f"不支持的编码器类型: {encoder_type}")


def _create_complete_model(udi, task_type, encoder_type, vocab_manager, pooling_method):
    """第1阶段：创建编码器，reset权重逻辑已在encoder创建时处理"""
    
    encoder_config = _build_encoder_config(udi.config, encoder_type, task_type)
    encoder = create_encoder(encoder_type, encoder_config, vocab_manager)
    
    weight_status = "已重置" if udi.config.reset_weights else "默认初始化"
    logger.info(f"🔧 编码器创建完成: {encoder_type}({encoder.get_hidden_size()}d), 权重{weight_status}")
    
    return encoder


def _load_and_copy_pretrained_weights(model, pretrained_path):
    """第2阶段：加载预训练encoder权重，覆盖第1阶段创建的权重"""
    
    pretrain_path = Path(pretrained_path)
    
    # 基础路径检查：使用断言，快速失败
    assert pretrain_path.exists(), f"预训练路径不存在: {pretrained_path}"
    assert (pretrain_path / 'config.bin').exists(), f"缺少预训练配置: {pretrain_path / 'config.bin'}"
    assert (pretrain_path / 'pytorch_model.bin').exists(), f"缺少预训练模型: {pretrain_path / 'pytorch_model.bin'}"
    
    # 直接加载状态字典，避免创建临时对象
    checkpoint = torch.load(pretrain_path / 'pytorch_model.bin', map_location='cpu')
    
    # 提取encoder权重
    encoder_state = {}
    for key, value in checkpoint.items():
        if key.startswith('encoder.'):
            encoder_key = key[8:]  # 移除 'encoder.' 前缀
            encoder_state[encoder_key] = value
    
    # 基础校验：使用断言
    assert encoder_state, "预训练模型中未找到encoder权重"
    
    # 检查词嵌入层权重（BERT使用bert.embeddings前缀）
    word_embedding_key = None
    for key in encoder_state.keys():
        if 'embeddings.word_embeddings.weight' in key:
            word_embedding_key = key
            break
    
    assert word_embedding_key is not None, f"缺少词嵌入层权重。可用键名: {list(encoder_state.keys())[:5]}..."
    
    # 兼容性检查
    pretrained_vocab_size = encoder_state[word_embedding_key].shape[0]
    
    # 检查位置嵌入大小
    position_embedding_key = None
    for key in encoder_state.keys():
        if 'position_embeddings.weight' in key:
            position_embedding_key = key
            break
    
    if position_embedding_key:
        pretrained_max_length = encoder_state[position_embedding_key].shape[0]
        
        # 获取当前模型的位置嵌入大小
        current_max_length = None
        if hasattr(model.encoder, 'bert'):
            current_max_length = model.encoder.bert.embeddings.position_embeddings.weight.shape[0]
            current_vocab_size = model.encoder.bert.get_input_embeddings().num_embeddings
        elif hasattr(model.encoder, 'gte_model'):
            current_max_length = model.encoder.gte_model.embeddings.position_embeddings.weight.shape[0]
            current_vocab_size = model.encoder.gte_model.get_input_embeddings().num_embeddings
        else:
            current_vocab_size = None
        
        # 兼容性检查和错误提示
        if current_vocab_size is not None:
            assert pretrained_vocab_size == current_vocab_size, \
                f"词表大小不一致：预训练={pretrained_vocab_size}, 当前={current_vocab_size}"
        
        if current_max_length and pretrained_max_length != current_max_length:
            logger.error("❌ 位置嵌入维度不兼容:")
            logger.error(f"  预训练模型: {pretrained_max_length}")
            logger.error(f"  当前模型: {current_max_length}")
            logger.error("💡 解决方案:")
            logger.error(f"  1. 微调时使用相同的max_seq_length: --max_seq_length {pretrained_max_length}")
            logger.error(f"  2. 或重新训练预训练模型使用max_seq_length={current_max_length}")
            raise ValueError(
                f"位置嵌入维度不匹配：预训练={pretrained_max_length}, 当前={current_max_length}。"
                f"请在微调时使用 --max_seq_length {pretrained_max_length} 或重新预训练。"
            )
        else:
            logger.info(f"✅ 兼容性检查通过: 词表={pretrained_vocab_size}, 位置嵌入={pretrained_max_length}")
    
    # 加载权重到当前encoder
    model.encoder.load_state_dict(encoder_state, strict=True)
    logger.info(f"✅ 预训练权重加载成功 ({len(encoder_state)}个参数)")
    logger.info("📝 最终权重状态: 预训练权重（已覆盖第1阶段权重）")


# 已删除旧的_copy_pretrained_weights函数，权重复制逻辑已内置到create_model_from_udi中