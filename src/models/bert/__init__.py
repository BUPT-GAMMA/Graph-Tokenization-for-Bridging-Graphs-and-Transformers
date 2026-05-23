"""
BERT training pipeline — Token ID sequence support.
BERT 训练 Pipeline - Token ID 序列支持

A BERT training system that takes token ID sequences as input:
支持 Token ID 序列输入的 BERT 训练系统，包含：
- Vocabulary management: auto-build vocab from token ID sequences / 词表管理
- MLM pre-training: Masked Language Modeling / MLM 预训练
- Regression tasks: sequence-level continuous prediction / 回归任务
- Full pipeline: end-to-end from data to model / 完整 pipeline

BERT 训练系统支持以下功能：
- 词表管理：自动从 Token ID 序列构建词表
- MLM 预训练：Masked Language Modeling
- 回归任务：序列级连续预测
- 完整 pipeline：从数据到模型的端到端支持
"""

from .vocab_manager import VocabManager, build_vocab_from_sequences
from .config import BertConfig  # 🆕 BertConfig从单独文件导入
from .data import (
    MLMDataset,
    create_mlm_dataloader
)

__version__ = "1.0.0"
__author__ = "Custom BERT Team"

__all__ = [
    # Vocabulary management / 词表管理
    "VocabManager", "build_vocab_from_sequences",
    
    # Configuration / 配置
    "BertConfig",
    
    # Data processing / 数据处理
    "MLMDataset",
    "create_mlm_dataloader",
    
    # Note: BertMLM has been migrated to src.models.universal_model.UniversalModel
    # 注意：BertMLM 已迁移到 src.models.universal_model.UniversalModel
] 