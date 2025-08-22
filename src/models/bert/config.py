"""
BERT配置类
==========

从原model.py中提取的BertConfig类，用于兼容性
"""

from __future__ import annotations


class BertConfig:
    """BERT配置"""
    
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
        """
        初始化BERT配置
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度，对应config.bert.architecture.hidden_size
            num_hidden_layers: BERT层数，对应config.bert.architecture.num_hidden_layers
            num_attention_heads: 注意力头数，对应config.bert.architecture.num_attention_heads
            intermediate_size: 前馈网络维度，对应config.bert.architecture.intermediate_size
            hidden_dropout_prob: 隐藏层dropout，对应config.bert.architecture.hidden_dropout_prob
            attention_probs_dropout_prob: 注意力dropout，对应config.bert.architecture.attention_probs_dropout_prob
            max_position_embeddings: 最大序列长度，对应config.bert.architecture.max_position_embeddings
            layer_norm_eps: 层归一化epsilon，对应config.bert.architecture.layer_norm_eps
            pad_token_id: padding token的ID，对应config.special_tokens.ids.pad
            dropout: dropout率，对应config.bert.architecture.dropout
            **kwargs: 其他配置参数
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
        
        # 额外配置
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_hf_config(self):
        """转换为HuggingFace BertConfig"""
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
