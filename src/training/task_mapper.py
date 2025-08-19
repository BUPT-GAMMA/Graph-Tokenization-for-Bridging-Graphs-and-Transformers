"""
任务类型映射和统一处理
======================

将复杂的任务类型映射到基础任务类型，统一处理不同数据集的任务需求。
"""

from typing import Dict, Any, Tuple, Literal, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 基础任务类型
BaseTaskType = Literal["regression", "classification"]
# 扩展任务类型  
ExtendedTaskType = Literal["regression", "classification", "multi_label_classification", "multi_target_regression"]


class TaskMapper:
    """任务类型映射器"""
    
    @staticmethod
    def map_to_base_task(task: ExtendedTaskType) -> BaseTaskType:
        """将扩展任务类型映射到基础任务类型"""
        mapping = {
            "regression": "regression",
            "multi_target_regression": "regression",
            "classification": "classification", 
            "multi_label_classification": "classification",
        }
        
        if task not in mapping:
            raise ValueError(f"不支持的任务类型: {task}")
            
        return mapping[task]
    
    @staticmethod
    def is_regression_task(task: ExtendedTaskType) -> bool:
        """判断是否为回归任务（包括多目标回归）"""
        return task in ["regression", "multi_target_regression"]
    
    @staticmethod
    def is_classification_task(task: ExtendedTaskType) -> bool:
        """判断是否为分类任务（包括多标签分类）"""
        return task in ["classification", "multi_label_classification"]
    
    @staticmethod
    def requires_normalizer(task: ExtendedTaskType) -> bool:
        """判断任务是否需要标签归一化器"""
        return TaskMapper.is_regression_task(task)
    
    @staticmethod
    def get_task_parameters(dataset_task_type: str, num_classes: int) -> Dict[str, Any]:
        """
        根据数据集任务类型获取任务参数
        
        Args:
            dataset_task_type: 数据集返回的任务类型（如get_dataset_task_type()）
            num_classes: 数据集返回的类别/目标数量
            
        Returns:
            任务参数字典
        """
        # 映射数据集任务类型到标准任务类型
        task_mapping = {
            "regression": ("regression", {"num_targets": 1}),
            "binary_classification": ("classification", {"num_classes": 2}),
            "classification": ("classification", {"num_classes": num_classes}),
            "multi_label_classification": ("multi_label_classification", {"num_labels": num_classes}),
            "multi_target_regression": ("multi_target_regression", {"num_targets": num_classes}),
        }
        
        if dataset_task_type not in task_mapping:
            raise ValueError(f"不支持的数据集任务类型: {dataset_task_type}")
        
        task, params = task_mapping[dataset_task_type]
        return {"task": task, **params}
    
    @staticmethod
    def get_loss_function(task: ExtendedTaskType, num_outputs: int):
        """获取适合的损失函数"""
        import torch.nn as nn
        
        if task == "regression":
            return nn.MSELoss()
        elif task == "multi_target_regression":
            return nn.L1Loss()  # 多目标回归用MAE
        elif task == "classification":
            return nn.CrossEntropyLoss()
        elif task == "multi_label_classification":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"不支持的任务类型: {task}")
    
    @staticmethod
    def get_primary_metric(task: ExtendedTaskType) -> str:
        """获取任务的主要评价指标"""
        metric_mapping = {
            "regression": "mae",
            "multi_target_regression": "macro_mae", 
            "classification": "roc_auc",  # molhiv用ROC-AUC
            "multi_label_classification": "macro_ap",  # peptides_func用AP
        }
        
        return metric_mapping.get(task, "loss")
    
    @staticmethod
    def should_maximize_metric(task: ExtendedTaskType) -> bool:
        """判断指标是否应该最大化（True）还是最小化（False）"""
        maximize_tasks = ["classification", "multi_label_classification"]
        return task in maximize_tasks


def auto_detect_task_config(dataset_name: str, config) -> Dict[str, Any]:
    """
    自动检测数据集的任务配置
    
    Args:
        dataset_name: 数据集名称
        config: 项目配置
        
    Returns:
        任务配置字典，包含task类型和相关参数
    """
    from src.data.unified_data_factory import get_dataloader
    
    loader = get_dataloader(dataset_name, config)
    dataset_task_type = loader.get_dataset_task_type()
    num_classes = loader.get_num_classes()
    
    task_config = TaskMapper.get_task_parameters(dataset_task_type, num_classes)
    
    logger.info(f"🎯 {dataset_name} 任务检测:")
    logger.info(f"  数据集任务类型: {dataset_task_type}")
    logger.info(f"  类别/目标数: {num_classes}")
    logger.info(f"  映射任务: {task_config['task']}")
    logger.info(f"  任务参数: {dict((k,v) for k,v in task_config.items() if k != 'task')}")
    
    return task_config
