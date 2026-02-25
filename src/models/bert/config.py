"""
BERT Configuration
BERT配置类

BertConfig extracted from the original model.py for backward compatibility.
从原model.py中提取的BertConfig类，用于兼容性。
"""

from __future__ import annotations


class BertConfig:
    """BERT configuration container.
    BERT配置容器。"""
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 layer_norm_eps: float = 1e-12,
                 pad_token_id: int = 0,
                 dropout: float = 0.1,
                 **kwargs):
        """Initialize BERT configuration.
        初始化BERT配置。
        
        Args:
            vocab_size: Vocabulary size / 词汇表大小
            hidden_size: Hidden dimension / 隐藏层维度 (config.bert.architecture.hidden_size)
            num_hidden_layers: Number of BERT layers / BERT层数
            num_attention_heads: Number of attention heads / 注意力头数
            intermediate_size: FFN dimension / 前馈网络维度
            hidden_dropout_prob: Hidden dropout / 隐藏层dropout
            attention_probs_dropout_prob: Attention dropout / 注意力dropout
            max_position_embeddings: Max sequence length / 最大序列长度
            layer_norm_eps: LayerNorm epsilon / 层归一化epsilon
            pad_token_id: Padding token ID / padding token的ID
            dropout: Dropout rate / dropout率
            **kwargs: Additional config params / 其他配置参数
        """
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        
        # Additional config / 额外配置
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_hf_config(self):
        """Convert to HuggingFace BertConfig.
        转换为HuggingFace BertConfig。"""
        from transformers import BertConfig as HFBertConfig
        return HFBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            position_embedding_type="absolute",
            use_cache=False
        )
