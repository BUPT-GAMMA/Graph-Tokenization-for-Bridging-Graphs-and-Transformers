#!/usr/bin/env python3
"""
测试统一任务系统
================

测试TaskMapper和统一的模型架构是否能正确处理所有任务类型。
"""

import torch
import numpy as np

from config import ProjectConfig
from src.training.task_mapper import TaskMapper, auto_detect_task_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_task_mapper():
    """测试TaskMapper功能"""
    logger.info("🔍 测试TaskMapper...")
    
    # 测试任务映射
    test_tasks = [
        "regression",
        "classification", 
        "multi_label_classification",
        "multi_target_regression"
    ]
    
    for task in test_tasks:
        try:
            base_task = TaskMapper.map_to_base_task(task)
            is_reg = TaskMapper.is_regression_task(task)
            is_cls = TaskMapper.is_classification_task(task)
            needs_norm = TaskMapper.requires_normalizer(task)
            primary_metric = TaskMapper.get_primary_metric(task)
            maximize = TaskMapper.should_maximize_metric(task)
            
            logger.info(f"  {task}:")
            logger.info(f"    基础任务: {base_task}")
            logger.info(f"    是否回归: {is_reg}, 是否分类: {is_cls}")
            logger.info(f"    需要归一化: {needs_norm}")
            logger.info(f"    主要指标: {primary_metric} ({'最大化' if maximize else '最小化'})")
            
        except Exception as e:
            logger.error(f"  ❌ {task} 测试失败: {e}")


def test_auto_detection():
    """测试自动任务检测"""
    logger.info("🎯 测试自动任务检测...")
    
    config = ProjectConfig()
    datasets = ["molhiv", "peptides_func", "peptides_struct"]
    
    for dataset_name in datasets:
        try:
            task_config = auto_detect_task_config(dataset_name, config)
            logger.info(f"  ✅ {dataset_name} 检测成功")
            
        except Exception as e:
            logger.error(f"  ❌ {dataset_name} 检测失败: {e}")


def main():
    """主函数"""
    logger.info("🚀 开始测试统一任务系统...")
    
    test_task_mapper()
    test_auto_detection()
    
    logger.info("🎉 统一任务系统测试完成!")


if __name__ == "__main__":
    main()
