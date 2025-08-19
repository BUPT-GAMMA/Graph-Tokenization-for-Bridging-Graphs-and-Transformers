#!/usr/bin/env python3
"""
测试多任务模型实现
==================

测试新增的多标签分类和多目标回归模型是否能正常工作。
"""

import torch
import numpy as np
from typing import Dict, Any

from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.models.bert.heads import create_task_head
from src.training.model_builder import load_pretrained_backbone
from src.utils.metrics import (
    compute_classification_metrics,
    compute_multi_label_classification_metrics, 
    compute_multi_target_regression_metrics
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_task_detection(config: ProjectConfig):
    """测试任务类型检测"""
    logger.info("🔍 测试任务类型检测...")
    
    datasets = ["molhiv", "peptides_func", "peptides_struct"]
    
    for dataset_name in datasets:
        try:
            loader = get_dataloader(dataset_name, config)
            task_type = loader.get_dataset_task_type()
            num_classes = loader.get_num_classes()
            
            logger.info(f"  {dataset_name}: {task_type}, 类别/目标数: {num_classes}")
            
            # 确定实际的任务参数
            if task_type == "binary_classification":
                task = "classification"
                task_params = {"num_classes": num_classes}
            elif task_type == "multi_label_classification": 
                task = "multi_label_classification"
                task_params = {"num_labels": num_classes}
            elif task_type == "multi_target_regression":
                task = "multi_target_regression" 
                task_params = {"num_targets": num_classes}
            else:
                logger.warning(f"  未知任务类型: {task_type}")
                continue
                
            logger.info(f"    → 映射任务: {task}, 参数: {task_params}")
            
        except Exception as e:
            logger.error(f"  {dataset_name} 检测失败: {e}")


def test_model_creation(config: ProjectConfig):
    """测试模型创建"""
    logger.info("🔧 测试模型创建...")
    
    # 创建虚拟的vocab_manager用于测试
    class MockVocabManager:
        def __init__(self):
            self.vocab_size = 1000
            self.pad_token_id = 0
    
    vocab_manager = MockVocabManager()
    
    # 测试各种任务头部
    test_cases = [
        ("classification", {"num_classes": 2}),
        ("multi_label_classification", {"num_labels": 10}),
        ("multi_target_regression", {"num_targets": 11}),
    ]
    
    for task, params in test_cases:
        try:
            model = create_task_head(
                task=task,
                vocab_manager=vocab_manager,
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=2048,
                pooling_method="mean",
                dropout=0.1,
                max_position_embeddings=768,
                layer_norm_eps=1e-12,
                **params
            )
            
            logger.info(f"  ✅ {task} 模型创建成功")
            
            # 测试前向传播
            batch_size = 4
            seq_len = 32
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            outputs = model(input_ids, attention_mask)
            logger.info(f"    输出形状: {outputs['logits'].shape if 'logits' in outputs else outputs['predictions'].shape}")
            
        except Exception as e:
            logger.error(f"  ❌ {task} 模型创建失败: {e}")


def test_metrics_computation():
    """测试评价指标计算"""
    logger.info("📊 测试评价指标计算...")
    
    # 测试多标签分类指标
    logger.info("  测试多标签分类指标...")
    n_samples, n_labels = 100, 10
    y_true_multi = np.random.randint(0, 2, (n_samples, n_labels))  # 二进制标签
    y_score_multi = np.random.rand(n_samples, n_labels)  # 概率分数
    
    try:
        metrics = compute_multi_label_classification_metrics(y_true_multi, y_score_multi)
        logger.info(f"    ✅ 多标签分类指标: macro_ap={metrics['macro_ap']:.4f}, exact_match={metrics['exact_match']:.4f}")
    except Exception as e:
        logger.error(f"    ❌ 多标签分类指标计算失败: {e}")
    
    # 测试多目标回归指标
    logger.info("  测试多目标回归指标...")
    n_samples, n_targets = 100, 11
    y_true_reg = np.random.randn(n_samples, n_targets)
    y_pred_reg = y_true_reg + np.random.randn(n_samples, n_targets) * 0.1  # 添加噪声
    
    try:
        metrics = compute_multi_target_regression_metrics(y_true_reg, y_pred_reg)
        logger.info(f"    ✅ 多目标回归指标: macro_mae={metrics['macro_mae']:.4f}, overall_mae={metrics['overall_mae']:.4f}")
    except Exception as e:
        logger.error(f"    ❌ 多目标回归指标计算失败: {e}")


def test_data_loading():
    """测试数据加载和标签格式"""
    logger.info("📂 测试数据加载...")
    
    config = ProjectConfig()
    
    datasets_config = [
        ("molhiv", "classification", "binary_classification"),
        ("peptides_func", "multi_label_classification", "multi_label_classification"), 
        ("peptides_struct", "multi_target_regression", "multi_target_regression"),
    ]
    
    for dataset_name, expected_task, loader_task_type in datasets_config:
        try:
            loader = get_dataloader(dataset_name, config)
            train_data, val_data, test_data, train_labels, val_labels, test_labels = loader.load_data()
            
            logger.info(f"  {dataset_name}:")
            logger.info(f"    数据加载: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
            logger.info(f"    任务类型: {loader.get_dataset_task_type()}")
            logger.info(f"    标签格式: {type(train_labels[0])}")
            
            if isinstance(train_labels[0], list):
                logger.info(f"    标签维度: {len(train_labels[0])}")
                logger.info(f"    标签示例: {train_labels[0]}")
            else:
                logger.info(f"    标签示例: {train_labels[0]}")
            
        except Exception as e:
            logger.error(f"  ❌ {dataset_name} 数据加载失败: {e}")


def main():
    """主函数"""
    logger.info("🚀 开始测试多任务模型实现...")
    
    config = ProjectConfig()
    
    # 运行各项测试
    test_task_detection(config)
    test_model_creation(config) 
    test_metrics_computation()
    test_data_loading()
    
    logger.info("🎉 多任务模型测试完成!")


if __name__ == "__main__":
    main()
