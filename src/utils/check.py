
from __future__ import annotations

from typing import List

import torch

from config import ProjectConfig

def check_vocab_compatibility(logger, token_sequences: List[List[int]], vocab_manager) -> None:
  
    if not vocab_manager:
        logger.warning("词表管理器未加载，跳过词表兼容性检查")
        return

    logger.info("🔍 检查词表兼容性...")
    all_tokens = set()
    for seq in token_sequences:
        all_tokens.update(seq)

    unknown_tokens = [t for t in all_tokens if t not in vocab_manager.token_to_id]
    total_tokens = len(all_tokens)
    unknown_count = len(unknown_tokens)
    unknown_ratio = (unknown_count / total_tokens * 100) if total_tokens > 0 else 0.0

    logger.info("📊 词表兼容性统计:")
    logger.info(f"   总token类型数: {total_tokens}")
    logger.info(f"   未知token类型数: {unknown_count}")
    logger.info(f"   未知token比例: {unknown_ratio:.2f}%")

    if unknown_count > 0:
        if unknown_ratio > 10:
            logger.error(f"❌ 未知token比例过高 ({unknown_ratio:.2f}%)！")
            logger.error("   建议使用与预训练相同的数据处理流程或重新预训练模型")
            logger.error(f"   未知token示例: {unknown_tokens[:10]}")
        elif unknown_ratio > 5:
            logger.warning(f"⚠️ 未知token比例较高 ({unknown_ratio:.2f}%)")
            logger.warning("   建议检查数据处理流程是否与预训练一致")
        else:
            logger.info(f"✅ 词表兼容性良好，未知token比例较低 ({unknown_ratio:.2f}%)")
    else:
        logger.info("✅ 词表完全兼容，所有token都在预训练词表中")
        

def infer_task_property(config: ProjectConfig, udi) -> tuple[str, int | None]:
    """
    推断任务类型和目标信息
    
    Args:
        config: 项目配置
        udi: 统一数据接口
        
    Returns:
        none
    """
    meta = udi.get_downstream_metadata()
    

    assert 'dataset_task_type' in meta, "数据集元数据中缺少必需字段 'dataset_task_type'"
    task = meta['dataset_task_type']
    
    # 处理回归任务的目标属性
    if task == 'regression' and not config.task.target_property:
        # QM9数据集默认使用homo属性
        if config.dataset.name.lower().startswith('qm9'):
            config.task.target_property = 'homo'
        else:
            # 其他数据集使用默认属性
            if 'default_target_property' in meta and meta['default_target_property']:
                config.task.target_property = meta['default_target_property']
        
        if config.task.target_property:
            print(f"🎯 自动设置回归目标属性: {config.task.target_property}")
    
    
        
def parse_torch_dtype(dtype_val):
    if dtype_val is None:
        return None
    if isinstance(dtype_val, torch.dtype):
        return dtype_val
    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return mapping.get(str(dtype_val).lower(), torch.float32)