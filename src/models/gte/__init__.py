"""
GTE (General Text Embeddings) 模型集成模块
===========================================

本模块提供GTE-multilingual-base模型的集成功能，
作为BERT模型的高性能替代方案。

主要特性：
- 高效的unpadding优化
- 内存高效的attention机制  
- 支持长序列处理（最大8192）
- 多语言支持
- 预训练模型直接微调

使用示例：
    from src.models.gte import example_gte_usage
    example_gte_usage()
"""

__version__ = "0.1.0"
__author__ = "TokenizerGraph Project"

# 导出主要功能（当实现后取消注释）
# from .model import GTETaskModel
# from .config import GTEConfig

__all__ = [
    # "GTETaskModel",
    # "GTEConfig", 
]
