
from __future__ import annotations

from typing import List
import logging
import numpy as np

logger = logging.getLogger(__name__)


def check_vocab_compatibility(token_sequences: List[List[int]], vocab_manager) -> None:
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