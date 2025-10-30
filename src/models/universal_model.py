"""
统一模型类
==========

UniversalModel - 支持所有任务类型的统一架构
- 预训练(MLM): 序列级处理，每个token位置预测
- 微调任务: 句子级处理，池化后预测

⚠️  重要：输出字段的语义与标准HuggingFace模型不同
================================================
标准HuggingFace BERT输出：
- last_hidden_state: [batch, seq_len, hidden] 每个token的表示
- pooler_output: [batch, hidden] 句子级表示

本项目UniversalModel输出（微调任务）：
- 'outputs': [batch, output_dim] ← 这是TaskHead(pooled)的结果，即最终任务预测
- 'pooled': [batch, hidden] ← 这是真正的句子级编码表示

关键区别：我们的'outputs'不是原始编码，而是经过任务头处理的最终预测！
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from src.utils.logger import get_logger

# 创建模块级logger
logger = get_logger(__name__)

from src.models.unified_encoder import BaseEncoder
from src.models.unified_task_head import UnifiedTaskHead


class UniversalModel(nn.Module):
    """统一模型 - 支持所有任务类型"""
    
    def __init__(
        self,
        encoder: BaseEncoder,
        task_type: str,
        output_dim: int,
        pooling_method: str = 'mean',
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.encoder = encoder
        self.task_type = task_type
        self.pooling_method = pooling_method
        task_head_config={'hidden_ratio': 0.5, 'activation': 'relu', 'dropout': 0.1}
        
        embedding_weight = None
        if task_type == 'mlm':
            embedding_weight = encoder.get_word_embeddings_weight()  # <- 这里拿到 [V,H]
            
        # 创建统一任务头
        self.task_head = UnifiedTaskHead(
            input_dim=encoder.get_hidden_size(),  # 编码器输出维度，如512或768
            task_type=task_type,
            output_dim=output_dim,                # 任务输出维度：MLM=vocab_size, 分类=num_classes
            config=task_head_config,
            embedding_weight=embedding_weight,
            dtype=dtype
        )
        
        # 保存元数据
        self.output_dim = output_dim
    
    def forward(
        self, 
        input_ids: torch.Tensor,          # [batch_size, seq_len] - token ID序列
        attention_mask: torch.Tensor,     # [batch_size, seq_len] - 注意力掩码，1=有效，0=pad
        labels: Optional[torch.Tensor] = None  # 标签，具体形状见下方注释
    ) -> Dict[str, torch.Tensor]:
        """
        统一前向传播 - 根据任务类型自动选择处理方式
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码
            labels: 标签张量，形状因任务而异：
                   - MLM: [batch_size, seq_len] 每个位置的目标token，-100表示不计算损失
                   - 分类: [batch_size] 类别索引
                   - 回归: [batch_size] 或 [batch_size, 1] 目标值
                   - 多目标回归: [batch_size, num_targets] 多个目标值
        
        Returns:
            字典包含以下键：
            - MLM任务: 
                * 'outputs': [batch_size, seq_len, vocab_size] 每个位置的词表概率
                * 'pooled': None (MLM不需要句子级表示)
            - 其他任务:
                * 'outputs': [batch_size, output_dim] 任务预测输出
                * 'pooled': [batch_size, hidden_size] 句子级编码表示
        
        ⚠️⚠️⚠️ 重要警告：语义与标准HuggingFace模型不同！
        对于微调任务：'outputs' = TaskHead('pooled')
        即outputs是pooled经过任务头处理的结果，二者不是平级关系！
        """
        
        if self.task_type == 'mlm':
            # MLM任务：序列级处理，每个token位置都要预测
            # 获取未池化的序列表示
            sequence_output = self.encoder.get_sequence_output(input_ids, attention_mask)
            # sequence_output: [batch_size, seq_len, hidden_size]
            
            # MLM预测头：线性投影到词表大小
            logits = self.task_head(sequence_output)
            # logits: [batch_size, seq_len, vocab_size]
            
            return {
                'outputs': logits,      # [batch_size, seq_len, vocab_size] - MLM预测logits
                'pooled': None          # MLM不需要池化表示
            }
        else:
            # 其他任务：句子级处理，需要将序列池化为单个向量
            # 获取池化后的句子表示
            pooled_output = self.encoder.encode(input_ids, attention_mask, self.pooling_method)
            # pooled_output: [batch_size, hidden_size]
            
            # 任务预测头：多层感知机
            logits = self.task_head(pooled_output)
            # logits: [batch_size, output_dim]
            
            # ⚠️⚠️⚠️ 重要：注意输出字段的依赖关系 ⚠️⚠️⚠️
            # 'outputs' = TaskHead('pooled')，即 outputs 是 pooled 的下游处理结果
            # 这与标准HuggingFace模型的输出语义不同！
            return {
                'outputs': logits,       # [batch_size, output_dim] - 最终任务预测（TaskHead的输出）
                'pooled': pooled_output  # [batch_size, hidden_size] - 原始句子编码表示（TaskHead的输入）
            }
    
    def predict(
        self, 
        input_ids: torch.Tensor,          # [batch_size, seq_len] token ID序列
        attention_mask: torch.Tensor      # [batch_size, seq_len] 注意力掩码
    ) -> torch.Tensor:
        """
        获取预测输出 - 兼容原有接口
        
        Returns:
            - MLM任务: [batch_size, seq_len, vocab_size] 词表预测概率
            - 其他任务: [batch_size, output_dim] 任务预测结果
        """
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            return result['outputs']  # 返回预测输出，形状见forward()注释
    
    def save_model(self, save_path: str):
        """保存统一模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        # print(list(self.state_dict().keys())[:10])
        
        # 保存配置信息
        config_to_save = {
            'task_type': self.task_type,
            'output_dim': self.output_dim,
            'pooling_method': self.pooling_method,
            'encoder_hidden_size': self.encoder.get_hidden_size()
        }
        torch.save(config_to_save, os.path.join(save_path, 'config.bin'))
        
        logger.info(f"🎯 UniversalModel已保存到: {save_path}")
     
    @classmethod
    def load_model(cls, model_path: str, encoder: BaseEncoder) -> 'UniversalModel':
        """加载统一模型"""
        import os
        
        # 加载配置
        config_data = torch.load(os.path.join(model_path, 'config.bin'), map_location='cpu')
        
        # 创建模型
        model = cls(
            encoder=encoder,
            task_type=config_data['task_type'],
            output_dim=config_data['output_dim'],
            pooling_method=config_data.get('pooling_method', 'mean')
        )
        
        # 加载权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"🎯 UniversalModel已从 {model_path} 加载完成")
        return model
