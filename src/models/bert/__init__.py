"""
BERT训练Pipeline - Token ID序列支持

这是一个支持Token ID序列输入的BERT训练系统，包含：
- 词表管理: 从Token ID序列自动构建词表
- MLM预训练: Masked Language Modeling预训练
- 回归任务: 序列级连续数值预测
- 完整pipeline: 从数据到模型的端到端流程
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
    # 词表管理
    "VocabManager", "build_vocab_from_sequences",
    
    # 配置
    "BertConfig",
    
    # 数据处理  
    "MLMDataset",
    "create_mlm_dataloader",
    
    # 注意：BertMLM已迁移到 src.models.universal_model.UniversalModel
] 