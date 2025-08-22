from __future__ import annotations


def load_pretrained_backbone(config, pretrained_dir=None):
    """统一的预训练模型加载接口，支持BERT和GTE等"""
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    
    # 使用新的配置项
    encoder_type = config.encoder.type
    logger.info(f"🔧 加载编码器类型: {encoder_type}")
    
    if encoder_type == 'bert':
        return _load_bert_backbone(config, pretrained_dir)
    elif 'gte' in encoder_type.lower():
        return _load_gte_backbone(config, pretrained_dir)
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")


def _load_bert_backbone(config, pretrained_dir=None):
    """加载BERT backbone - 重构为UniversalModel加载"""
    from pathlib import Path
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    # 🆕 现在返回UniversalModel而不是BertMLM
    # 如果有预训练模型，尝试加载为UniversalModel
    if pretrained_dir is not None:
        p = Path(pretrained_dir)
        if (p / 'config.bin').exists() and (p / 'pytorch_model.bin').exists():
            try:
                # TODO: 实现UniversalModel加载
                logger.warning("⚠️ UniversalModel加载暂未实现，使用随机初始化")
            except Exception:
                logger.warning("⚠️ 预训练模型格式不兼容，使用随机初始化")
        else:
            logger.warning(f"⚠️ 预训练目录无效: {pretrained_dir}")

    # 🔧 简化：直接返回None，让create_model_from_udi处理
    # 这样可以避免复杂的预训练模型格式转换
    logger.warning("⚠️ 未找到预训练BERT模型，将使用随机初始化")
    return None


def _load_gte_backbone(config, pretrained_dir=None):
    """加载GTE backbone"""
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("🚀 创建预训练GTE编码器...")
    
    # 获取vocab_manager
    from src.data.unified_data_interface import UnifiedDataInterface
    udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)
    vocab_manager = udi.get_vocab(method=config.serialization.method)
    
    # 构建GTE配置
    gte_config = {
        'hidden_size': 768,  # GTE固定768维
        'max_seq_length': 8192,  # GTE支持长序列
        'vocab_size': vocab_manager.vocab_size,
        'optimization': {
            'unpad_inputs': True,
            'use_memory_efficient_attention': True,
            'torch_dtype': 'float32'  # 微调时使用float32
        }
    }
    
    # 创建GTE编码器（使用HuggingFace预训练权重）
    from src.models.unified_encoder import create_encoder
    gte_encoder = create_encoder(
        model_name=config.encoder.type,  # 'Alibaba-NLP/gte-multilingual-base'
        config=gte_config,
        vocab_manager=vocab_manager
    )
    
    logger.info(f"✅ GTE编码器创建完成: {gte_encoder.get_hidden_size()}维")
    return gte_encoder


def build_task_model(
    config,
    udi,
    method,
    pretrained_dir=None,
    pretrain_exp_name=None,
):
    """
    构建统一任务模型 - 完全重构版
    
    Args:
        config: 项目配置
        udi: 统一数据接口
        method: 序列化方法
        pretrained_dir: 预训练模型目录（可选）
        pretrain_exp_name: 预训练实验名（可选）
        
    Returns:
        (model, task_handler) 元组
    """
    from src.models.bert.heads import create_model_from_udi
    
    # 🆕 预训练模型加载逻辑内置到create_model_from_udi中
    # 通过pretrain_exp_name实现灵活的预训练模型指定
    if pretrain_exp_name is not None:
        # 临时修改配置中的实验名，用于预训练模型查找
        original_exp_name = config.experiment_name
        config.experiment_name = pretrain_exp_name
        
        try:
            model, task_handler = create_model_from_udi(
                udi, 
                pretrained_model=None,  # 自动加载
                pooling_method=config.bert.architecture.pooling_method
            )
            return model, task_handler
        finally:
            # 恢复原始实验名
            config.experiment_name = original_exp_name
    else:
        # 标准流程
        model, task_handler = create_model_from_udi(
            udi, 
            pretrained_model=None,
            pooling_method=config.bert.architecture.pooling_method
        )
        
        return model, task_handler





