#!/usr/bin/env python3
"""
测试统一模型系统
================

验证BertUnified + TaskHandler的组合能正确处理所有任务类型。
"""

import torch
import numpy as np
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.models.bert.heads import create_model_from_udi
from src.models.bert.pretrained_manager import get_pretrained
from src.training.task_handler import TaskHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_dataset_with_unified_model(dataset_name: str, config: ProjectConfig):
    """测试单个数据集的统一模型处理"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 测试数据集: {dataset_name}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. 创建数据加载器
        udi = get_dataloader(dataset_name, config)
        task_type = udi.get_dataset_task_type()
        num_classes = udi.get_num_classes()
        
        logger.info(f"  任务类型: {task_type}")
        logger.info(f"  类别/目标数: {num_classes}")
        
        # 2. 获取预训练模型（mock）
        class MockPretrained:
            def __init__(self):
                from src.models.bert.config import BertConfig
                from src.models.bert.vocab import VocabManager
                
                # Mock vocab manager
                self.vocab_manager = VocabManager()
                self.vocab_manager.vocab_size = 1000
                self.vocab_manager.pad_token_id = 0
                
                # Mock config
                self.config = BertConfig(
                    vocab_size=1000,
                    hidden_size=256,
                    num_hidden_layers=2,
                    num_attention_heads=8,
                    intermediate_size=1024,
                    dropout=0.1,
                    max_position_embeddings=128,
                    layer_norm_eps=1e-12,
                    pad_token_id=0
                )
                
                # Mock BERT model
                from transformers import BertModel
                hf_config = self.config.to_hf_config()
                self.bert = BertModel(hf_config)
        
        pretrained = MockPretrained()
        
        # 3. 创建统一模型和任务处理器
        model, task_handler = create_model_from_udi(
            udi=udi,
            pretrained_model=pretrained,
            pooling_method='mean'
        )
        
        logger.info(f"  ✅ 模型创建成功")
        logger.info(f"  输出维度: {task_handler.output_dim}")
        logger.info(f"  主要指标: {task_handler.primary_metric}")
        logger.info(f"  需要归一化: {task_handler.requires_normalizer}")
        
        # 4. 测试前向传播
        batch_size = 4
        seq_len = 32
        
        # 创建模拟输入
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # 创建模拟标签
        if task_type == "regression":
            labels = torch.randn(batch_size, 1)
        elif task_type == "multi_target_regression":
            labels = torch.randn(batch_size, num_classes)
        elif task_type in ["binary_classification", "classification"]:
            labels = torch.randint(0, num_classes, (batch_size,))
        elif task_type == "multi_label_classification":
            labels = torch.randint(0, 2, (batch_size, num_classes)).float()
        else:
            raise ValueError(f"未知任务类型: {task_type}")
        
        # 5. 测试模型输出
        with torch.no_grad():
            # 不带标签的前向传播
            outputs = model(input_ids, attention_mask)
            logger.info(f"  模型输出形状: {outputs['outputs'].shape}")
            
            # 测试损失计算
            loss = task_handler.compute_loss(outputs['outputs'], labels)
            logger.info(f"  损失值: {loss.item():.4f}")
            
            # 测试预测
            predictions = task_handler.get_predictions(outputs['outputs'])
            logger.info(f"  预测形状: {predictions.shape}")
            
            # 测试概率（仅分类任务）
            probs = task_handler.get_probabilities(outputs['outputs'])
            if probs is not None:
                logger.info(f"  概率形状: {probs.shape}")
        
        logger.info(f"  ✅ {dataset_name} 测试通过!")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ {dataset_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始测试统一模型系统...")
    
    config = ProjectConfig()
    
    # 测试所有支持的数据集
    test_datasets = [
        "molhiv",           # binary_classification
        "peptides_func",    # multi_label_classification  
        "peptides_struct",  # multi_target_regression
    ]
    
    results = {}
    for dataset in test_datasets:
        results[dataset] = test_dataset_with_unified_model(dataset, config)
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("📊 测试结果汇总:")
    logger.info(f"{'='*60}")
    
    for dataset, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"  {dataset}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 所有测试通过! 统一模型系统工作正常。")
    else:
        logger.info("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
