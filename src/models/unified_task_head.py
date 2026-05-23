"""
Unified Task Head Manager
统一任务头管理器

UnifiedTaskHead — builds different prediction heads based on task type.
根据任务类型构建不同结构的预测头。
- MLM: simple linear layer, sequence-level output / 简单线性层，序列级输出
- Others: MLP, sentence-level output / 多层感知机，句子级输出
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

# Module-level logger / 模块级logger
logger = get_logger(__name__)

# class MLMHead(nn.Module):
#     """
#     BERT-MLM标准头：
#       transform: Linear(H,H) + GELU + LayerNorm
#       decoder  : 权重与 word_embeddings.weight 绑定 + 独立 bias
#     输入: [B, T, H] -> 输出: [B, T, V]
#     """
#     def __init__(self, hidden_size, vocab_size, embedding_weight, layer_norm_eps=1e-12, dtype=torch.float32):
#         super().__init__()
#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.act   = nn.GELU()
#         self.ln    = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
#         self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
#         # 绑定外部 embedding 的权重张量（[V, H]）
#         self.embedding_weight = embedding_weight
#         self.dtype = dtype
#         # BERT 风格初始化
#         nn.init.normal_(self.dense.weight, mean=0.0, std=0.02)
#         nn.init.zeros_(self.dense.bias)
#         nn.init.zeros_(self.decoder_bias)

#     def forward(self, hidden_states):  # [B,T,H]
#         x = self.ln(self.act(self.dense(hidden_states)))
#         W = self.embedding_weight.to(x.dtype)      # [V,H]
#         b = self.decoder_bias.to(x.dtype)          # [V]
#         return F.linear(x, W, b)                   # [B,T,V]

class UnifiedTaskHead(nn.Module):
    """Unified task head manager — builds different prediction head structures based on task type.
    统一任务头管理器 - 根据任务类型构建不同结构的预测头。"""
    
    def __init__(
        self, 
        input_dim: int,     # Encoder output dim, e.g. 512(BERT-Small) or 768(BERT-Base/GTE) / 编码器输出维度
        task_type: str,     # Task type: 'mlm', 'classification', 'regression', etc. / 任务类型
        output_dim: int,    # Output dim: MLM=vocab_size, cls=num_classes, reg=1 or num_targets / 输出维度
        config: Dict, # Task head config params / 任务头配置参数
        embedding_weight: torch.Tensor=None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.task_type = task_type
        self.output_dim = output_dim
        self.input_dim = input_dim
        assert input_dim is not None and input_dim > 0, "输入维度不能为空"
        assert output_dim is not None and output_dim > 0, "输出维度不能为空"
        
        
        if task_type == 'mlm':
            # MLM task: simple linear projection, no complex structure needed
            # MLM任务：简单线性投影，不需要复杂结构
            # input: [batch_size, seq_len, hidden_size] → output: [batch_size, seq_len, vocab_size]
            assert embedding_weight is not None, "embedding_weight不能为空"
            self.head = nn.Linear(input_dim, output_dim, dtype=dtype)  # hidden_size → vocab_size
# self.head = MLMHead(input_dim, output_dim, embedding_weight, dtype=dtype)  # hidden_size → vocab_size
            logger.info(f"🔤 MLM任务头: Linear({input_dim} → {output_dim})")
        else:
            # Other tasks: MLP, supports more complex feature transforms
            # 其他任务：多层感知机，支持更复杂的特征变换
            # input: [batch_size, hidden_size] → output: [batch_size, output_dim]
            self.head = self._build_configurable_head(input_dim, output_dim, config, dtype)
            logger.info(f"🎯 {task_type}任务头: MLP({input_dim} → {config['hidden_ratio']} → {config['activation']} → {config['dropout']} → {output_dim})")
        
        # Initialize weights / 初始化权重
        self._init_weights()
    
    def _build_configurable_head(self, input_dim: int, output_dim: int, config: Dict, dtype: torch.dtype):
        """Build a configurable multi-layer task head.
        构建可配置的多层任务头。
        
        Args:
            input_dim: Input dimension [hidden_size] / 输入维度
            output_dim: Output dimension [num_classes or 1] / 输出维度
            config: Config dict with ['hidden_ratio', 'activation', 'dropout'] / 配置字典
            
        Returns:
            nn.Sequential: MLP structure / 多层感知机结构
        """
        
        # Parse config params with sensible defaults / 解析配置参数
        hidden_ratio = config['hidden_ratio']      # Hidden layer size ratio / 隐藏层大小比例
        activation = config['activation']       # Activation function type / 激活函数类型
        dropout = config['dropout']               # Dropout ratio / dropout比例
        
        layers = []
        
        # First layer: input -> hidden / 第一层：输入层 → 隐藏层
        hidden_dim = int(input_dim * hidden_ratio)  # 如512*0.5=256
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
        # Linear: [batch_size, input_dim] → [batch_size, hidden_dim]
        
        # Activation function / 激活函数
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        # Dropout regularization / Dropout正则化
        layers.append(nn.Dropout(dropout))
        
        # Output layer: hidden -> output / 输出层：隐藏层 → 输出维度
        layers.append(nn.Linear(hidden_dim, output_dim, dtype=dtype))
        # Linear: [batch_size, hidden_dim] → [batch_size, output_dim]
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize task head weights using standard initialization.
        初始化任务头权重 - 使用标准初始化策略。"""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                # Weights: normal distribution init / 权重：正态分布初始化
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — handles different input shapes based on task type.
        前向传播 - 根据任务类型处理不同形状的输入。
        
        Args:
            x: Input tensor, shape varies by task / 输入张量，形状因任务而异：
               - MLM: [batch_size, seq_len, hidden_size] sequence-level features / 序列级特征
               - Others: [batch_size, hidden_size] sentence-level features / 句子级特征
               
        Returns:
            torch.Tensor: Prediction output / 预测输出，形状：
               - MLM: [batch_size, seq_len, vocab_size] per-position vocab logits / 每个位置的词表预测
               - Classification: [batch_size, num_classes] class logits / 类别logits
               - Regression: [batch_size, 1] or [batch_size, num_targets] / 回归值
        """
        
        if self.task_type == 'mlm':
            # MLM input: [batch_size, seq_len, hidden_size]
            # MLM output: [batch_size, seq_len, vocab_size]
            # Predict vocab distribution for each position / 对序列的每个位置进行词表预测
            if x.dim() != 3:
                raise ValueError(f"MLM任务期望3维输入 [batch, seq_len, hidden]，实际获得{x.dim()}维: {x.shape}")
            
            batch_size, seq_len, hidden_size = x.shape
            assert hidden_size == self.input_dim, f"输入维度不匹配：期望{self.input_dim}，实际{hidden_size}"
            
            logits = self.head(x)  # [batch_size, seq_len, vocab_size]
            expected_shape = (batch_size, seq_len, self.output_dim)
            assert logits.shape == expected_shape, \
                f"MLM输出形状不匹配：期望{expected_shape}，实际{logits.shape}"
            
            return logits
        else:
            # Other task input: [batch_size, hidden_size]
            # Other task output: [batch_size, output_dim]
            if x.dim() != 2:
                raise ValueError(f"{self.task_type}任务期望2维输入 [batch, hidden]，实际获得{x.dim()}维: {x.shape}")
            
            batch_size, hidden_size = x.shape
            assert hidden_size == self.input_dim, f"输入维度不匹配：期望{self.input_dim}，实际{hidden_size}"
            
            logits = self.head(x)  # [batch_size, output_dim]
            expected_shape = (batch_size, self.output_dim)
            assert logits.shape == expected_shape, \
                f"{self.task_type}输出形状不匹配：期望{expected_shape}，实际{logits.shape}"
            
            return logits
