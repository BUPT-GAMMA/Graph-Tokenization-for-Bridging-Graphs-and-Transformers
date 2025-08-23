from __future__ import annotations
from pathlib import Path

from src.models.bert.heads import create_model_from_udi
from src.utils.logger import get_logger

# 创建模块级logger
logger = get_logger(__name__)


def build_task_model(
    config,
    udi,
    method,
    pretrained_dir=None,
    pretrain_exp_name=None,
):
    """
    构建统一任务模型 - 重新设计版本，逻辑清晰简洁
    
    Args:
        config: 项目配置
        udi: 统一数据接口
        method: 序列化方法
        pretrained_dir: 预训练模型目录（可选）
        pretrain_exp_name: 预训练实验名（可选）
        
    Returns:
        (model, task_handler) 元组
    """
    logger.info("🚀 构建任务模型...")
    logger.info(f"  数据集: {config.dataset.name}")
    logger.info(f"  序列化方法: {method}")
    logger.info(f"  编码器类型: {config.encoder.type}")
    
    # 🆕 内置路径解析逻辑
    pretrained_path = _resolve_pretrained_path_internal(config, pretrain_exp_name, pretrained_dir)
    
    if pretrained_path:
        logger.info(f"📦 将使用预训练模型: {pretrained_path}")
    else:
        logger.info("🆕 将创建新模型（无预训练权重）")
    
    return create_model_from_udi(udi, pretrained_path)


def _resolve_pretrained_path_internal(config, pretrain_exp_name, pretrained_dir):
    """内部化的预训练路径解析，避免创建额外文件"""
    
    def _validate_model_dir(path):
        """验证模型目录是否包含必需文件"""
        required_files = ['config.bin', 'pytorch_model.bin']
        return all((path / f).exists() for f in required_files)
    
    # 1. 显式预训练目录优先（最高优先级）
    if pretrained_dir is not None:
        logger.debug(f"检查显式预训练目录: {pretrained_dir}")
        p = Path(pretrained_dir)
        if p.exists() and _validate_model_dir(p):
            return str(p)
        logger.warning(f"⚠️ 指定的预训练目录无效: {pretrained_dir}")
        return None
    
    # 2. 使用pretrain_exp_name（中等优先级）  
    if pretrain_exp_name is not None:
        logger.debug(f"使用预训练实验名搜索: {pretrain_exp_name}")
        pretrain_path = config.get_model_dir().parent / pretrain_exp_name / config.dataset.name / config.serialization.method
        
        for subdir in ['best', 'final']:  # best优先
            candidate = pretrain_path / subdir
            if candidate.exists() and _validate_model_dir(candidate):
                logger.info(f"✅ 从指定预训练实验找到模型: {pretrain_exp_name} -> {candidate} 作为预训练权重")
                return str(candidate)
        
        logger.info(f"⚠️ 从pretrain_exp_name: {pretrain_exp_name} 中未找到有效模型")
    
    # 3. 使用当前experiment_name（最低优先级）
    base_dir = config.get_model_dir()
    for subdir in ['best', 'final']:  # best优先
        candidate = base_dir / subdir
        if candidate.exists() and _validate_model_dir(candidate):
            logger.info(f"✅ 采用experiment_name: {config.experiment_name} 找到模型")
            return str(candidate)
    
    return None





