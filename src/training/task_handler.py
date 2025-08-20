"""
任务处理器
==========

处理不同任务类型的损失函数、后处理和指标计算。
保持简洁，避免过度设计。
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TaskHandler:
    """
    处理不同任务类型的逻辑
    
    核心职责：
    1. 根据任务类型选择合适的损失函数
    2. 处理模型输出的后处理（如softmax、sigmoid）
    3. 提供任务特定的预测方法
    """
    
    def __init__(self, task_type: str, output_dim: int):
        """
        Args:
            task_type: 任务类型（从UDI获取）
            output_dim: 输出维度
        """
        self.task_type = task_type
        self.output_dim = output_dim
        self.loss_fn = self._get_loss_function()
    
    def is_regression_task(self) -> bool:
        """判断是否为回归任务"""
        return self.task_type in ["regression", "multi_target_regression"]
    
    def is_classification_task(self) -> bool:
        """判断是否为分类任务"""
        return self.task_type in ["binary_classification", "classification", "multi_label_classification"]
    
    def is_multi_label(self) -> bool:
        """判断是否为多标签分类任务"""
        return self.task_type == "multi_label_classification"
    
    def is_multi_target(self) -> bool:
        """判断是否为多目标回归任务"""
        return self.task_type == "multi_target_regression"
    
    def _get_loss_function(self):
        """根据任务类型返回损失函数"""
        if self.task_type == "regression":
            # 单目标回归：MSE
            return nn.MSELoss()
        elif self.task_type == "multi_target_regression":
            # 多目标回归：MAE（L1Loss）
            return nn.L1Loss()
        elif self.task_type == "binary_classification":
            # 二分类：CrossEntropy
            return nn.CrossEntropyLoss()
        elif self.task_type == "classification":
            # 多分类：CrossEntropy
            return nn.CrossEntropyLoss()
        elif self.task_type == "multi_label_classification":
            # 多标签分类：BCEWithLogitsLoss
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    def compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            outputs: 模型输出 [batch_size, output_dim]
            labels: 真实标签
        
        Returns:
            损失值
        """
        # 根据任务类型调整标签格式
        if self.task_type == "regression":
            # 单目标回归：确保标签是[batch_size, 1]
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            loss = self.loss_fn(outputs, labels.float())
            
        elif self.task_type == "multi_target_regression":
            # 多目标回归：标签已经是[batch_size, num_targets]
            loss = self.loss_fn(outputs, labels.float())
            
        elif self.task_type in ["binary_classification", "classification"]:
            # 分类：标签是整数索引
            loss = self.loss_fn(outputs, labels.long())
            
        elif self.task_type == "multi_label_classification":
            # 多标签分类：标签是二进制向量
            loss = self.loss_fn(outputs, labels.float())
            
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
        
        return loss
    
    def process_outputs(
        self,
        outputs: torch.Tensor,
        return_probs: bool = False
    ) -> torch.Tensor:
        """
        处理模型输出，返回预测值或概率
        
        Args:
            outputs: 模型原始输出
            return_probs: 是否返回概率（仅对分类任务有效）
        
        Returns:
            处理后的输出
        """
        with torch.no_grad():
            if self.task_type in ["regression", "multi_target_regression"]:
                # 回归：直接返回原始输出
                return outputs
                
            elif self.task_type in ["binary_classification", "classification"]:
                if return_probs:
                    # 返回softmax概率
                    return torch.softmax(outputs, dim=-1)
                else:
                    # 返回类别索引
                    return torch.argmax(outputs, dim=-1)
                    
            elif self.task_type == "multi_label_classification":
                if return_probs:
                    # 返回sigmoid概率
                    return torch.sigmoid(outputs)
                else:
                    # 返回二进制预测（阈值0.5）
                    return (torch.sigmoid(outputs) > 0.5).float()
                    
            else:
                raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    def get_predictions(
        self,
        outputs: torch.Tensor
    ) -> np.ndarray:
        """
        获取最终预测值（用于评估）
        
        Args:
            outputs: 模型输出
            
        Returns:
            numpy格式的预测值
        """
        with torch.no_grad():
            if self.task_type in ["regression", "multi_target_regression"]:
                # 回归：直接返回
                return outputs.cpu().numpy()
                
            elif self.task_type in ["binary_classification", "classification"]:
                # 分类：返回预测类别
                return torch.argmax(outputs, dim=-1).cpu().numpy()
                
            elif self.task_type == "multi_label_classification":
                # 多标签：返回二进制预测
                return (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                
            else:
                raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    def get_probabilities(
        self,
        outputs: torch.Tensor
    ) -> Optional[np.ndarray]:
        """
        获取概率（仅对分类任务）
        
        Args:
            outputs: 模型输出
            
        Returns:
            概率数组，回归任务返回None
        """
        with torch.no_grad():
            if self.task_type in ["regression", "multi_target_regression"]:
                return None
                
            elif self.task_type in ["binary_classification", "classification"]:
                return torch.softmax(outputs, dim=-1).cpu().numpy()
                
            elif self.task_type == "multi_label_classification":
                return torch.sigmoid(outputs).cpu().numpy()
                
            else:
                raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    @property
    def primary_metric(self) -> str:
        """获取主要评价指标"""
        metric_map = {
            "regression": "mae",
            "multi_target_regression": "macro_mae",
            "binary_classification": "roc_auc",
            "classification": "accuracy",
            "multi_label_classification": "ap"
        }
        return metric_map.get(self.task_type, "loss")
    
    @property
    def should_maximize_metric(self) -> bool:
        """判断主要指标是否应该最大化"""
        maximize_tasks = [
            "binary_classification",
            "classification",
            "multi_label_classification"
        ]
        return self.task_type in maximize_tasks
    
    @property
    def requires_normalizer(self) -> bool:
        """判断是否需要标签归一化"""
        return self.task_type in ["regression", "multi_target_regression"]


def create_task_handler(udi) -> TaskHandler:
    """
    从UnifiedDataInterface创建任务处理器
    
    Args:
        udi: UnifiedDataInterface实例
        
    Returns:
        TaskHandler实例
    """
    task_type = udi.get_dataset_task_type()
    
    # 获取输出维度
    if task_type in ["regression"]:
        output_dim = 1
    elif task_type in ["binary_classification"]:
        output_dim = 2
    elif task_type in ["classification", "multi_label_classification", "multi_target_regression"]:
        output_dim = udi.get_num_classes()
    else:
        raise ValueError(f"无法确定输出维度: {task_type}")
    
    logger.info(f"📋 创建任务处理器: {task_type} (输出维度={output_dim})")
    
    return TaskHandler(task_type, output_dim)
