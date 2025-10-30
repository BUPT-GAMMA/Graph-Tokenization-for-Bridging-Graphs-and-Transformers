"""
统一任务头管理器
================

UnifiedTaskHead - 根据任务类型构建不同结构的预测头
- MLM任务: 简单线性层，序列级输出
- 其他任务: 多层感知机，句子级输出
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
<<<<<<< HEAD

=======
import torch.nn.functional as F

from src.utils.logger import get_logger

# 创建模块级logger
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
>>>>>>> dev

class UnifiedTaskHead(nn.Module):
    """统一任务头管理器 - 根据任务类型构建不同结构的预测头"""
    
    def __init__(
        self, 
        input_dim: int,     # 编码器输出维度，如512(BERT-Small)或768(BERT-Base/GTE)
        task_type: str,     # 任务类型：'mlm', 'classification', 'regression'等
        output_dim: int,    # 输出维度：MLM=vocab_size, 分类=num_classes, 回归=1或num_targets
<<<<<<< HEAD
        config: Dict = None # 任务头配置参数
=======
        config: Dict, # 任务头配置参数
        embedding_weight: torch.Tensor=None,
        dtype: torch.dtype = torch.float32
>>>>>>> dev
    ):
        super().__init__()
        
        self.task_type = task_type
        self.output_dim = output_dim
        self.input_dim = input_dim
<<<<<<< HEAD
=======
        assert input_dim is not None and input_dim > 0, "输入维度不能为空"
        assert output_dim is not None and output_dim > 0, "输出维度不能为空"
        
>>>>>>> dev
        
        if task_type == 'mlm':
            # MLM任务：简单线性投影，不需要复杂结构
            # input: [batch_size, seq_len, hidden_size] → output: [batch_size, seq_len, vocab_size]
<<<<<<< HEAD
            self.head = nn.Linear(input_dim, output_dim)  # hidden_size → vocab_size
            print(f"🔤 MLM任务头: Linear({input_dim} → {output_dim})")
        else:
            # 其他任务：多层感知机，支持更复杂的特征变换
            # input: [batch_size, hidden_size] → output: [batch_size, output_dim]
            self.head = self._build_configurable_head(input_dim, output_dim, config or {})
            print(f"🎯 {task_type}任务头: MLP({input_dim} → ... → {output_dim})")
=======
            assert embedding_weight is not None, "embedding_weight不能为空"
            self.head = nn.Linear(input_dim, output_dim, dtype=dtype)  # hidden_size → vocab_size
# self.head = MLMHead(input_dim, output_dim, embedding_weight, dtype=dtype)  # hidden_size → vocab_size
            logger.info(f"🔤 MLM任务头: Linear({input_dim} → {output_dim})")
        else:
            # 其他任务：多层感知机，支持更复杂的特征变换
            # input: [batch_size, hidden_size] → output: [batch_size, output_dim]
            self.head = self._build_configurable_head(input_dim, output_dim, config, dtype)
            logger.info(f"🎯 {task_type}任务头: MLP({input_dim} → {config['hidden_ratio']} → {config['activation']} → {config['dropout']} → {output_dim})")
>>>>>>> dev
        
        # 初始化权重
        self._init_weights()
    
<<<<<<< HEAD
    def _build_configurable_head(self, input_dim: int, output_dim: int, config: Dict):
=======
    def _build_configurable_head(self, input_dim: int, output_dim: int, config: Dict, dtype: torch.dtype):
>>>>>>> dev
        """
        构建可配置的多层任务头
        
        Args:
            input_dim: 输入维度 [hidden_size]
            output_dim: 输出维度 [num_classes或1]
<<<<<<< HEAD
            config: 配置字典，包含hidden_ratio, activation, dropout等
=======
            config: 配置字典，包含['hidden_ratio', 'activation', 'dropout']
>>>>>>> dev
            
        Returns:
            nn.Sequential: 多层感知机结构
        """
        
        # 解析配置参数，提供合理默认值
<<<<<<< HEAD
        hidden_ratio = config.get('hidden_ratio', 0.5)      # 隐藏层大小比例
        activation = config.get('activation', 'relu')       # 激活函数类型
        dropout = config.get('dropout', 0.1)               # dropout比例
=======
        hidden_ratio = config['hidden_ratio']      # 隐藏层大小比例
        activation = config['activation']       # 激活函数类型
        dropout = config['dropout']               # dropout比例
>>>>>>> dev
        
        layers = []
        
        # 第一层：输入层 → 隐藏层
        hidden_dim = int(input_dim * hidden_ratio)  # 如512*0.5=256
<<<<<<< HEAD
        layers.append(nn.Linear(input_dim, hidden_dim))
=======
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
>>>>>>> dev
        # 线性层: [batch_size, input_dim] → [batch_size, hidden_dim]
        
        # 激活函数
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        # Dropout正则化
        layers.append(nn.Dropout(dropout))
        
        # 输出层：隐藏层 → 输出维度
<<<<<<< HEAD
        layers.append(nn.Linear(hidden_dim, output_dim))
=======
        layers.append(nn.Linear(hidden_dim, output_dim, dtype=dtype))
>>>>>>> dev
        # 线性层: [batch_size, hidden_dim] → [batch_size, output_dim]
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化任务头权重 - 使用标准初始化策略"""
<<<<<<< HEAD
        for module in self.head.modules() if hasattr(self.head, 'modules') else [self.head]:
            if isinstance(module, nn.Linear):
                # 权重：正态分布初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # 偏置：零初始化
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                # 权重：正态分布初始化
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
>>>>>>> dev
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 根据任务类型处理不同形状的输入
        
        Args:
            x: 输入张量，形状因任务而异：
               - MLM: [batch_size, seq_len, hidden_size] 序列级特征
               - 其他: [batch_size, hidden_size] 句子级特征
               
        Returns:
            torch.Tensor: 预测输出，形状：
               - MLM: [batch_size, seq_len, vocab_size] 每个位置的词表预测
               - 分类: [batch_size, num_classes] 类别logits
               - 回归: [batch_size, 1] 或 [batch_size, num_targets] 回归值
        """
        
        if self.task_type == 'mlm':
            # MLM输入: [batch_size, seq_len, hidden_size]
            # MLM输出: [batch_size, seq_len, vocab_size]
            # 对序列的每个位置进行词表预测
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
            # 其他任务输入: [batch_size, hidden_size]
            # 其他任务输出: [batch_size, output_dim]
            if x.dim() != 2:
                raise ValueError(f"{self.task_type}任务期望2维输入 [batch, hidden]，实际获得{x.dim()}维: {x.shape}")
            
            batch_size, hidden_size = x.shape
            assert hidden_size == self.input_dim, f"输入维度不匹配：期望{self.input_dim}，实际{hidden_size}"
            
            logits = self.head(x)  # [batch_size, output_dim]
            expected_shape = (batch_size, self.output_dim)
            assert logits.shape == expected_shape, \
                f"{self.task_type}输出形状不匹配：期望{expected_shape}，实际{logits.shape}"
            
            return logits
