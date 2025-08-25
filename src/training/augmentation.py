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
from typing import Dict, Tuple, Optional


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
        if not self.should_use_gaussian_noise():
            return embeddings
            
        if random.random() > self.aug_config.gaussian_noise_probability:
            return embeddings
            
        noise = torch.randn_like(embeddings) * self.aug_config.gaussian_noise_std
        return embeddings + noise
    
    def prepare_feature_mixup_batch(self, batch1: Dict, batch2: Dict) -> Tuple[Dict, float]:
        """准备特征混合的batch对"""
        if not self.should_use_feature_mixup():
            return batch1, 0.0
            
        if random.random() > self.aug_config.feature_mixup_probability:
            return batch1, 0.0
            
        # Beta分布采样混合系数
        alpha = self.aug_config.feature_mixup_alpha
        lam = np.random.beta(alpha, alpha)
        
        # 创建混合batch（只混合标签，特征在模型内部混合）
        mixed_batch = {
            'input_ids1': batch1['input_ids'],
            'attention_mask1': batch1['attention_mask'],
            'labels1': batch1['labels'],
            'input_ids2': batch2['input_ids'],
            'attention_mask2': batch2['attention_mask'], 
            'labels2': batch2['labels'],
            'mixup_lambda': lam
        }
        
        return mixed_batch, lam
    
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


def create_augmentation(config, task_type: str = "auto") -> Optional[TrainingAugmentation]:
    """创建增强器（如果配置启用）"""
    aug = TrainingAugmentation(config, task_type)
    
    # 检查是否需要任何增强
    if (aug.should_use_gaussian_noise() or 
        aug.should_use_feature_mixup() or 
        aug.should_use_consistency_regularization()):
        return aug
    
    return None
