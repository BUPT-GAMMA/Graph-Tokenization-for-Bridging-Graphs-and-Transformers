"""
Universal Model
统一模型类

UniversalModel — a unified architecture supporting all task types.
支持所有任务类型的统一架构。
- Pre-training (MLM): sequence-level, per-token prediction / 预训练: 序列级处理，每个token位置预测
- Fine-tuning: sentence-level, pooled prediction / 微调任务: 句子级处理，池化后预测

⚠️  Important: output field semantics differ from standard HuggingFace models
重要：输出字段的语义与标准HuggingFace模型不同
================================================
Standard HuggingFace BERT output / 标准HuggingFace BERT输出：
- last_hidden_state: [batch, seq_len, hidden] per-token representation / 每个token的表示
- pooler_output: [batch, hidden] sentence-level representation / 句子级表示

This project's UniversalModel output (fine-tuning) / 本项目UniversalModel输出（微调任务）：
- 'outputs': [batch, output_dim] ← TaskHead(pooled) result, i.e. final task prediction / 最终任务预测
- 'pooled': [batch, hidden] ← actual sentence-level encoded representation / 真正的句子级编码表示

Key difference: our 'outputs' is NOT raw encoding, but the final prediction after the task head!
关键区别：我们的'outputs'不是原始编码，而是经过任务头处理的最终预测！
"""

from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn

from src.utils.logger import get_logger

# Module-level logger / 模块级logger
logger = get_logger(__name__)

from src.models.unified_encoder import BaseEncoder
from src.models.unified_task_head import UnifiedTaskHead


class UniversalModel(nn.Module):
    """Universal model — supports all task types.
    统一模型 - 支持所有任务类型。"""
    
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
            embedding_weight = encoder.get_word_embeddings_weight()  # <- get [V,H] here
            
        # Create unified task head / 创建统一任务头
        self.task_head = UnifiedTaskHead(
            input_dim=encoder.get_hidden_size(),  # Encoder output dim, e.g. 512 or 768 / 编码器输出维度
            task_type=task_type,
            output_dim=output_dim,                # Task output dim: MLM=vocab_size, cls=num_classes / 任务输出维度
            config=task_head_config,
            embedding_weight=embedding_weight,
            dtype=dtype
        )
        
        # Save metadata / 保存元数据
        self.output_dim = output_dim
    
    def forward(
        self, 
        input_ids: torch.Tensor,          # [batch_size, seq_len] - token ID sequence / token ID序列
        attention_mask: torch.Tensor,     # [batch_size, seq_len] - attention mask, 1=valid, 0=pad / 注意力掩码
        labels: Optional[torch.Tensor] = None  # Labels, shape varies by task / 标签
    ) -> Dict[str, torch.Tensor]:
        """Unified forward pass — automatically selects processing based on task type.
        统一前向传播 - 根据任务类型自动选择处理方式。
        
        Args:
            input_ids: [batch_size, seq_len] token ID sequence / token ID序列
            attention_mask: [batch_size, seq_len] attention mask / 注意力掩码
            labels: Label tensor, shape varies by task / 标签张量，形状因任务而异：
                   - MLM: [batch_size, seq_len] target tokens per position, -100=ignore / 每个位置的目标token
                   - Classification: [batch_size] class indices / 分类: 类别索引
                   - Regression: [batch_size] or [batch_size, 1] target values / 回归: 目标值
                   - Multi-target regression: [batch_size, num_targets] / 多目标回归
        
        Returns:
            Dict containing / 字典包含以下键：
            - MLM task / MLM任务: 
                * 'outputs': [batch_size, seq_len, vocab_size] per-position vocab logits / 每个位置的词表概率
                * 'pooled': None (MLM doesn't need sentence-level repr / MLM不需要句子级表示)
            - Other tasks / 其他任务:
                * 'outputs': [batch_size, output_dim] task prediction / 任务预测输出
                * 'pooled': [batch_size, hidden_size] sentence-level encoding / 句子级编码表示
        
        ⚠️⚠️⚠️ Warning: semantics differ from standard HuggingFace models!
        重要警告：语义与标准HuggingFace模型不同！
        For fine-tuning: 'outputs' = TaskHead('pooled')
        对于微调任务：outputs是pooled经过任务头处理的结果，二者不是平级关系！
        """
        
        if self.task_type == 'mlm':
            # MLM task: sequence-level, every token position needs prediction
            # MLM任务：序列级处理，每个token位置都要预测
            # Get unpooled sequence representation / 获取未池化的序列表示
            sequence_output = self.encoder.get_sequence_output(input_ids, attention_mask)
            # sequence_output: [batch_size, seq_len, hidden_size]
            
            # MLM prediction head: linear projection to vocab size / MLM预测头：线性投影到词表大小
            logits = self.task_head(sequence_output)
            # logits: [batch_size, seq_len, vocab_size]
            
            return {
                'outputs': logits,      # [batch_size, seq_len, vocab_size] - MLM prediction logits / MLM预测logits
                'pooled': None          # MLM doesn't need pooled repr / MLM不需要池化表示
            }
        else:
            # Other tasks: sentence-level, pool sequence into a single vector
            # 其他任务：句子级处理，需要将序列池化为单个向量
            # Get pooled sentence representation / 获取池化后的句子表示
            pooled_output = self.encoder.encode(input_ids, attention_mask, self.pooling_method)
            # pooled_output: [batch_size, hidden_size]
            
            # Task prediction head: MLP / 任务预测头：多层感知机
            logits = self.task_head(pooled_output)
            # logits: [batch_size, output_dim]
            
            # ⚠️⚠️⚠️ Important: note output field dependency ⚠️⚠️⚠️
            # 'outputs' = TaskHead('pooled'), i.e. outputs is the downstream result of pooled
            # 重要：outputs 是 pooled 的下游处理结果，这与标准HuggingFace模型的输出语义不同！
            return {
                'outputs': logits,       # [batch_size, output_dim] - final task prediction (TaskHead output) / 最终任务预测
                'pooled': pooled_output  # [batch_size, hidden_size] - raw sentence encoding (TaskHead input) / 原始句子编码表示
            }
    
    def predict(
        self, 
        input_ids: torch.Tensor,          # [batch_size, seq_len] token ID sequence / token ID序列
        attention_mask: torch.Tensor      # [batch_size, seq_len] attention mask / 注意力掩码
    ) -> torch.Tensor:
        """Get prediction output — backward-compatible interface.
        获取预测输出 - 兼容原有接口。
        
        Returns:
            - MLM: [batch_size, seq_len, vocab_size] vocab prediction logits / 词表预测概率
            - Others: [batch_size, output_dim] task prediction / 任务预测结果
        """
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
            return result['outputs']  # Return prediction output, see forward() for shapes / 返回预测输出
    
    def save_model(self, save_path: str):
        """Save unified model.
        保存统一模型。"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model weights / 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        # print(list(self.state_dict().keys())[:10])
        
        # Save config info / 保存配置信息
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
        """Load unified model.
        加载统一模型。"""
        import os
        
        # Load config / 加载配置
        config_data = torch.load(os.path.join(model_path, 'config.bin'), map_location='cpu')
        
        # Create model / 创建模型
        model = cls(
            encoder=encoder,
            task_type=config_data['task_type'],
            output_dim=config_data['output_dim'],
            pooling_method=config_data.get('pooling_method', 'mean')
        )
        
        # Load weights / 加载权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"🎯 UniversalModel已从 {model_path} 加载完成")
        return model
