"""
统一的BERT预测模型
==================

一个真正统一的模型，支持所有预测任务。
输出维度由任务决定，损失函数和指标在外部处理。
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import BertModel

from src.models.bert import BertConfig
from src.models.bert import VocabManager


class BertUnified(nn.Module):
    """
    统一的BERT模型 - 不区分任务类型
    
    核心理念：
    - 模型只负责：BERT编码 → 池化 → 线性变换 → N维输出
    - 输出维度由外部决定（1维单回归、2维二分类、N维多分类/多回归）
    - 损失函数、指标计算等任务相关逻辑在外部处理
    """
    
    def __init__(
        self,
        config: BertConfig,
        vocab_manager: VocabManager,
        output_dim: int,
        pooling_method: str = 'mean'
    ):
        super().__init__()
        
        self.config = config
        self.vocab_manager = vocab_manager
        self.output_dim = output_dim
        self.pooling_method = pooling_method
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # 统一的预测头：输出指定维度
        self.prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.prediction_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _pool_sequence(
        self,
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """序列池化"""
        if self.pooling_method == 'mean':
            # 平均池化 (忽略padding)
            lengths = attention_mask.sum(dim=1, keepdim=True).float()
            masked_output = sequence_output * attention_mask.unsqueeze(-1)
            pooled = masked_output.sum(dim=1) / lengths
            return pooled
        elif self.pooling_method == 'cls':
            # 使用[CLS] token
            return sequence_output[:, 0, :]
        elif self.pooling_method == 'max':
            # 最大池化
            masked_output = sequence_output.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )
            pooled = masked_output.max(dim=1)[0]
            return pooled
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码
            labels: [batch_size, output_dim] 或 [batch_size] 标签
                    （损失计算由外部处理）
            
        Returns:
            包含以下键的字典：
            - outputs: [batch_size, output_dim] 原始输出（logits或predictions）
            - pooled: [batch_size, hidden_size] 池化后的表示
            - loss: 如果提供labels，由外部TaskHandler计算
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 序列池化
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = self._pool_sequence(sequence_output, attention_mask)
        
        # 预测头输出
        outputs = self.prediction_head(pooled_output)
        
        result = {
            'outputs': outputs,  # 统一命名为outputs
            'pooled': pooled_output
        }
        
        # 注意：损失计算交给外部的TaskHandler处理
        # 这里不计算损失，保持模型的纯粹性
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """获取原始预测值"""
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            return result['outputs']
    
    def save_model(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), 
                  os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存配置
        config_to_save = {
            'config': self.config,
            'output_dim': self.output_dim,
            'pooling_method': self.pooling_method
        }
        torch.save(config_to_save,
                  os.path.join(save_path, 'config.bin'))
        
        print(f"统一模型已保存到: {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str, vocab_manager: VocabManager) -> 'BertUnified':
        """加载模型"""
        import os
        
        # 加载配置
        config_data = torch.load(
            os.path.join(model_path, 'config.bin'),
            map_location='cpu'
        )
        
        # 创建模型
        model = cls(
            config=config_data['config'],
            vocab_manager=vocab_manager,
            output_dim=config_data['output_dim'],
            pooling_method=config_data.get('pooling_method', 'mean')
        )
        
        # 加载权重
        state_dict = torch.load(
            os.path.join(model_path, 'pytorch_model.bin'),
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        
        print(f"统一模型已从 {model_path} 加载")
        return model
