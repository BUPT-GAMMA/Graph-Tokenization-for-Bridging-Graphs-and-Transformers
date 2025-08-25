"""
任务处理器
==========

处理不同任务类型的损失函数、后处理和指标计算。
保持简洁，避免过度设计。
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def __init__(self, task_type: str, output_dim: int, dataset_name: str = None, vocab_size: int = None):
        """
        Args:
            task_type: 任务类型（从UDI获取或手动指定）
            output_dim: 输出维度
            dataset_name: 数据集名称（用于特定数据集的指标配置）
            vocab_size: 词表大小（MLM任务需要）
        """
        self.task_type = task_type
        self.output_dim = output_dim
        self.dataset_name = dataset_name
        self.vocab_size = vocab_size  # 🆕 MLM任务需要
        
        # 🆕 MLM任务时自动设置输出维度
        if task_type == 'mlm' and vocab_size is not None:
            self.output_dim = vocab_size
            
        self.loss_fn = self._get_loss_function()
    
    def is_mlm_task(self) -> bool:
        """判断是否为MLM预训练任务"""
        return self.task_type == "mlm"
    
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
        if self.task_type == "mlm":
            # 🆕 MLM任务：CrossEntropy with ignore_index=-100 (与原BertMLM一致)
            return nn.CrossEntropyLoss(ignore_index=-100)
        elif self.task_type == "regression":
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
            outputs: 模型输出，形状因任务而异：
                    - MLM: [batch_size, seq_len, vocab_size] MLM logits
                    - 其他: [batch_size, output_dim] 任务预测输出
            labels: 真实标签，形状因任务而异：
                   - MLM: [batch_size, seq_len] 每个位置的目标token，-100表示不计算损失
                   - 分类: [batch_size] 类别索引
                   - 回归: [batch_size] 或 [batch_size, num_targets] 目标值
        
        Returns:
            损失值 (标量)
        """
        # 根据任务类型调整损失计算方式
        if self.task_type == "mlm":
            # 🆕 MLM任务：特殊的序列级损失计算 (与原BertMLM.forward()完全一致)
            # outputs: [batch_size, seq_len, vocab_size]
            # labels: [batch_size, seq_len] 
            if self.vocab_size is None:
                raise ValueError("MLM任务需要提供vocab_size")
            loss = self.loss_fn(
                outputs.view(-1, self.vocab_size),  # [batch_size*seq_len, vocab_size]
                labels.view(-1)                     # [batch_size*seq_len]
            )
            
        elif self.task_type == "regression":
            # 单目标回归：确保标签是[batch_size, 1]
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            # 与输出保持相同dtype，避免与bf16混用导致的dtype错误
            labels = labels.to(outputs.dtype)
            loss = self.loss_fn(outputs, labels)
            
        elif self.task_type == "multi_target_regression":
            # 多目标回归：标签已经是[batch_size, num_targets]
            labels = labels.to(outputs.dtype)
            loss = self.loss_fn(outputs, labels)
            
        elif self.task_type in ["binary_classification", "classification"]:
            # 分类：标签是整数索引
            loss = self.loss_fn(outputs, labels.long())
            
        elif self.task_type == "multi_label_classification":
            # 多标签分类：标签是二进制向量
            labels = labels.to(outputs.dtype)
            loss = self.loss_fn(outputs, labels)
            
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
        
        return loss
    
    def compute_consistency_loss(
        self,
        outputs1: torch.Tensor,
        outputs2: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性正则化损失（R-Drop）
        
        Args:
            outputs1: 第一次前向传播的输出
            outputs2: 第二次前向传播的输出（dropout不同）
            labels: 真实标签
            
        Returns:
            一致性损失值
        """
        if self.task_type == "mlm":
            # MLM任务：计算vocabulary分布的KL散度
            logits1 = outputs1.view(-1, self.vocab_size)  # [batch*seq, vocab]
            logits2 = outputs2.view(-1, self.vocab_size)
            
            # 创建mask，只对非padding位置计算KL散度
            mask = (labels.view(-1) != -100)  # [batch*seq]
            
            if mask.sum() == 0:  # 如果没有有效位置，返回0
                return torch.tensor(0.0, device=outputs1.device)
                
            # 只对有效位置计算KL散度
            p = F.softmax(logits1[mask], dim=-1)
            q = F.softmax(logits2[mask], dim=-1)
            
            # 双向KL散度并平均
            kl_loss = (F.kl_div(F.log_softmax(logits1[mask], dim=-1), q, reduction='batchmean') +
                      F.kl_div(F.log_softmax(logits2[mask], dim=-1), p, reduction='batchmean')) / 2
            
        elif self.task_type in ["regression", "multi_target_regression"]:
            # 回归任务：直接计算输出的MSE
            kl_loss = F.mse_loss(outputs1, outputs2)
            
        elif self.task_type in ["binary_classification", "classification", "multi_label_classification"]:
            # 分类任务：计算logits的KL散度
            p = F.softmax(outputs1, dim=-1)
            q = F.softmax(outputs2, dim=-1)
            
            # 双向KL散度并平均
            kl_loss = (F.kl_div(F.log_softmax(outputs1, dim=-1), q, reduction='batchmean') +
                      F.kl_div(F.log_softmax(outputs2, dim=-1), p, reduction='batchmean')) / 2
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
            
        return kl_loss
    
    def compute_loss_with_consistency(
        self,
        outputs1: torch.Tensor,
        outputs2: torch.Tensor,
        labels: torch.Tensor,
        consistency_alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算包含一致性正则化的总损失
        
        Args:
            outputs1: 第一次前向传播的输出
            outputs2: 第二次前向传播的输出
            labels: 真实标签
            consistency_alpha: 一致性损失的权重
            
        Returns:
            (总损失, 任务损失, 一致性损失)
        """
        # 计算标准任务损失（两次输出的平均）
        task_loss1 = self.compute_loss(outputs1, labels)
        task_loss2 = self.compute_loss(outputs2, labels)
        task_loss = (task_loss1 + task_loss2) / 2
        
        # 计算一致性损失
        consistency_loss = self.compute_consistency_loss(outputs1, outputs2, labels)
        
        # 总损失
        total_loss = task_loss + consistency_alpha * consistency_loss
        
        return total_loss, task_loss, consistency_loss
    
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
                # 回归：直接返回（bfloat16 先转 float32 再 numpy）
                return outputs.detach().to(torch.float32).cpu().numpy()
                
            elif self.task_type in ["binary_classification", "classification"]:
                # 分类：返回预测类别（long → numpy 支持）
                return torch.argmax(outputs, dim=-1).detach().cpu().numpy()
                
            elif self.task_type == "multi_label_classification":
                # 多标签：返回二进制预测
                return (torch.sigmoid(outputs) > 0.5).float().detach().cpu().numpy()
                
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
                # 概率在 bf16 下先转 float32 再 numpy
                return torch.softmax(outputs, dim=-1).to(torch.float32).detach().cpu().numpy()
                
            elif self.task_type == "multi_label_classification":
                return torch.sigmoid(outputs).to(torch.float32).detach().cpu().numpy()
                
            else:
                raise ValueError(f"不支持的任务类型: {self.task_type}")
    
    @property
    def primary_metric(self) -> str:
        """获取主要评价指标"""
        # 基本指标映射
        metric_map = {
            "mlm": "loss",  # 🆕 MLM任务使用loss作为主要指标
            "regression": "mae",
            "multi_target_regression": "mae",  # 多目标回归也使用mae（平均计算）
            "binary_classification": "roc_auc",
            "classification": "accuracy",  # 默认用accuracy
            "multi_label_classification": "ap"
        }
        
        # 特殊数据集的指标配置
        if self.task_type == "classification" and self.dataset_name == "molhiv":
            return "roc_auc"  # molhiv特例使用AUC
        
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
    
    def compute_metrics(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        计算主要评价指标 - 使用现有的指标计算函数
        
        Args:
            outputs: 模型输出
            labels: 真实标签
            
        Returns:
            包含主要指标的字典
        """
        from src.utils.metrics import (
            compute_regression_metrics,
            compute_classification_metrics, 
            compute_multi_label_classification_metrics,
            compute_multi_target_regression_metrics
        )
        
        with torch.no_grad():
            if self.task_type == "mlm":
                # MLM任务：计算困惑度
                # outputs: [batch, seq_len, vocab_size], labels: [batch, seq_len]
                valid_positions = (labels != -100)
                if valid_positions.sum() > 0:
                    valid_outputs = outputs[valid_positions]  # [valid_positions, vocab_size]
                    valid_labels = labels[valid_positions]    # [valid_positions]
                    
                    log_probs = torch.log_softmax(valid_outputs, dim=-1)
                    nll = -log_probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1)
                    perplexity = torch.exp(nll.mean()).item()
                    
                    return {"perplexity": perplexity}
                else:
                    return {"perplexity": float('inf')}
                    
            elif self.task_type == "regression":
                # 🆕 使用现有的回归指标计算函数
                y_true = labels.cpu().numpy().flatten()
                y_pred = outputs.cpu().numpy().flatten()
                return compute_regression_metrics(y_true, y_pred)
                
            elif self.task_type == "multi_target_regression":
                # 🆕 使用现有的多目标回归指标计算函数
                y_true = labels.cpu().numpy()
                y_pred = outputs.cpu().numpy()
                return compute_multi_target_regression_metrics(y_true, y_pred)
                
            elif self.task_type in ["binary_classification", "classification"]:
                # 🆕 使用现有的分类指标计算函数
                y_true = labels.cpu().numpy()
                y_pred = self.get_predictions(outputs)  # 获取预测类别
                y_score = self.get_probabilities(outputs)  # 获取概率
                return compute_classification_metrics(y_true, y_pred, y_score=y_score)
                
            elif self.task_type == "multi_label_classification":
                # 🆕 使用现有的多标签分类指标计算函数
                y_true = labels.cpu().numpy()
                y_score = self.get_probabilities(outputs)  # 获取概率
                return compute_multi_label_classification_metrics(y_true, y_score)
                
            else:
                return {}


def create_task_handler(udi, task_type: str , vocab_size: int = None) -> Tuple[TaskHandler, int]:
    """
    创建任务处理器 - 支持MLM、强制任务类型和UDI推断
    
    Args:
        udi: UnifiedDataInterface实例（可选）
        task_type: 任务类型（可选，强制指定时使用）
        vocab_size: 词表大小（MLM任务需要）
        
    Returns:
        TaskHandler实例
    """
    
    # 模式1: MLM任务（预训练使用）
    if task_type == 'mlm':
        assert vocab_size is not None, "MLM任务需要提供vocab_size"
        
        logger.info(f"📋 创建MLM任务处理器: vocab_size={vocab_size}")
        return TaskHandler(task_type='mlm', output_dim=vocab_size, vocab_size=vocab_size), vocab_size
    
    # 模式2: 根据上层指定任务类型
    if task_type is not None and task_type != 'mlm':
        # 硬编码输出维度，与heads.py保持一致
        if task_type == 'regression':
            output_dim = 1
        elif task_type == 'binary_classification':
            output_dim = 2
        elif task_type in ['classification', 'multi_label_classification', 'multi_target_regression']:
            # 这种情况需要UDI来获取类别数
            if udi is None:
                raise ValueError(f"任务类型 {task_type} 需要UDI来确定输出维度")
            output_dim = udi.get_num_classes()
        else:
            raise ValueError(f"不支持的强制任务类型: {task_type}")
        
        dataset_name = udi.dataset 
        logger.info(f"📋 创建任务处理器: {task_type} (dataset={dataset_name}, output_dim={output_dim})")
        return TaskHandler(task_type, output_dim, dataset_name), output_dim
    
    # 模式3: 从UDI推断任务类型（标准微调使用）
    raise ValueError("流程异常：上层未指定task_type，将根据UDI推断任务类型(理论上这一步推断应该在上层完成)")
    
