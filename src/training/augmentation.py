"""
训练时数据增强工具
==================

简洁的增强实现，通过配置控制。
设计原则：不使用时对代码侵入性最小。

支持的增强方法：
1. 一致性正则化（R-Drop）：对同一输入两次前向传播，惩罚输出差异 [推荐]
2. 高斯噪声：在特征表示上添加噪声
3. 特征混合（Feature Mixup）：在特征空间混合两个样本 [仅回归任务]

配置方法：
在config/default_config.yml中设置：
```yaml
augmentation_config:
  # R-Drop（推荐优先启用）
  use_consistency_regularization: true
  consistency_alpha: 1.0
  
  # 高斯噪声
  use_gaussian_noise: false
  gaussian_noise_std: 0.01
  gaussian_noise_probability: 0.3
  
  # 特征混合（仅回归任务）
  use_feature_mixup: false
  feature_mixup_alpha: 0.2
  feature_mixup_probability: 0.2
```

使用方式：修改配置文件即可，无需修改训练代码。
"""

import random
import torch
import numpy as np
from typing import Optional


class TrainingAugmentation:
    """训练时增强的统一接口"""
    
    def __init__(self, config, task_type: str = "auto"):
        self.config = config
        
        # 预训练和微调使用不同配置
        if task_type == "pretraining" or task_type == "mlm":
            self.aug_config = config.bert.pretraining.augmentation_config
        elif task_type == "finetuning" or task_type == "regression" or task_type == "classification":
            self.aug_config = config.bert.finetuning.augmentation_config
        elif task_type == "auto":
            # 自动检测：优先检查是否有微调配置的特有字段
            if (hasattr(config.bert, 'finetuning') and 
                hasattr(config.bert.finetuning, 'augmentation_config') and
                hasattr(config.bert.finetuning.augmentation_config, 'sequence_masking_probability')):
                self.aug_config = config.bert.finetuning.augmentation_config
            elif hasattr(config.bert, 'pretraining'):
                self.aug_config = config.bert.pretraining.augmentation_config
            else:
                self.aug_config = None
        else:
            self.aug_config = None
            
    def should_use_gaussian_noise(self) -> bool:
        """是否使用高斯噪声"""
        return self.aug_config and self.aug_config.use_gaussian_noise
        
    def should_use_feature_mixup(self) -> bool:
        """是否使用特征混合"""
        return self.aug_config and self.aug_config.use_feature_mixup
        
    def should_use_consistency_regularization(self) -> bool:
        """是否使用一致性正则化"""
        return self.aug_config and self.aug_config.use_consistency_regularization
    
    def apply_gaussian_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """对嵌入添加高斯噪声"""
        if random.random() > self.aug_config.gaussian_noise_probability:
            return embeddings
            
        noise = torch.randn_like(embeddings) * self.aug_config.gaussian_noise_std
        return embeddings + noise
    
    def should_do_mixup_this_step(self) -> bool:
        """判断这一步是否要做特征混合"""
        return random.random() <= self.aug_config.feature_mixup_probability
    
    def get_mixup_lambda(self) -> float:
        """生成mixup的混合系数（调用前已确定要做mixup）"""
        alpha = self.aug_config.feature_mixup_alpha
        return np.random.beta(alpha, alpha)
    
    def mix_features(self, features1: torch.Tensor, features2: torch.Tensor, 
                    lam: float) -> torch.Tensor:
        """在特征空间进行混合"""
        return lam * features1 + (1 - lam) * features2
    
    def mix_labels(self, labels1: torch.Tensor, labels2: torch.Tensor, 
                  lam: float, task_type: str) -> torch.Tensor:
        """混合标签（只对回归任务有意义）"""
        if task_type in ["regression", "multi_target_regression"]:
            return lam * labels1 + (1 - lam) * labels2
        else:
            # 分类任务：随机选择一个标签
            mask = torch.rand(labels1.shape[0], device=labels1.device) < lam
            return torch.where(mask, labels1, labels2)
    
    def compute_training_loss(self, model, batch, task_handler, prev_batch=None):
        """计算训练损失（流水式处理所有增强逻辑）"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'] 
        labels = batch['labels']
        
        # 步骤1：确定增强策略
        use_mixup = (self.should_use_feature_mixup() and 
                    task_handler.is_regression_task() and 
                    self.should_do_mixup_this_step() and 
                    prev_batch is not None)
        use_consistency = self.should_use_consistency_regularization()
        use_noise = (self.should_use_gaussian_noise() and model.task_type != 'mlm')
        
        # 步骤2：获取特征和标签
        if use_mixup:
            # 特征混合路径
            prev_input_ids = prev_batch['input_ids'].to(input_ids.device)
            prev_attention_mask = prev_batch['attention_mask'].to(input_ids.device)
            prev_labels = prev_batch['labels'].to(input_ids.device)
            
            outputs1 = model(input_ids, attention_mask)
            outputs2 = model(prev_input_ids, prev_attention_mask)
            
            # 混合特征和标签
            lam = self.get_mixup_lambda()
            final_features = self.mix_features(outputs1['pooled'], outputs2['pooled'], lam)
            final_labels = self.mix_labels(labels, prev_labels, lam, task_handler.task_type)
        else:
            # 标准路径
            outputs = model(input_ids, attention_mask)
            final_features = outputs.get('pooled')
            final_labels = labels
            
        # 步骤3：应用高斯噪声
        if use_noise and final_features is not None:
            final_features = self.apply_gaussian_noise(final_features)
            
        # 步骤4：计算损失
        if use_consistency:
            if use_mixup:
                # Mixup + Consistency组合：
                # 1. 已经得到mixed_features和mixed_labels
                # 2. 用mixed_features通过task_head做两次前向传播（利用task_head的dropout）
                outputs1 = model.task_head(final_features)
                outputs2 = model.task_head(final_features)
                
                # 返回总损失
                total_loss, _, _ = task_handler.compute_loss_with_consistency(
                    outputs1, outputs2, final_labels, 
                    self.aug_config.consistency_alpha
                )
                return total_loss
            else:
                # 纯R-Drop：对整个模型做两次完整前向传播
                outputs1 = model(input_ids, attention_mask)
                outputs2 = model(input_ids, attention_mask)
                
                # 如果启用高斯噪声，在每次前向传播后单独应用
                if use_noise and outputs1.get('pooled') is not None:
                    # 第一次前向传播加噪声
                    noisy_pooled1 = self.apply_gaussian_noise(outputs1['pooled'])
                    outputs1['outputs'] = model.task_head(noisy_pooled1)
                    
                    # 第二次前向传播加噪声
                    noisy_pooled2 = self.apply_gaussian_noise(outputs2['pooled'])
                    outputs2['outputs'] = model.task_head(noisy_pooled2)
                
                # 返回总损失
                total_loss, _, _ = task_handler.compute_loss_with_consistency(
                    outputs1['outputs'], outputs2['outputs'], final_labels, 
                    self.aug_config.consistency_alpha
                )
                return total_loss
        else:
            # 标准损失计算
            if use_mixup or use_noise:
                # 使用处理过的特征
                final_outputs = model.task_head(final_features)
            else:
                # 使用原始输出
                final_outputs = outputs['outputs']
                
            return task_handler.compute_loss(final_outputs, final_labels)


def create_augmentation(config, task_type: str = "auto") -> Optional[TrainingAugmentation]:
    """创建增强器（如果配置启用）"""
    aug = TrainingAugmentation(config, task_type)
    
    # 检查是否需要任何增强
    if (aug.should_use_gaussian_noise() or 
        aug.should_use_feature_mixup() or 
        aug.should_use_consistency_regularization()):
        return aug
    
    return None
