#!/usr/bin/env python3
"""
测试统一模型的微调流程
======================

遵循项目标准的finetune_pipeline流程，测试BertUnified + TaskHandler。
"""

import torch
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.models.bert.heads import create_model_from_udi
from src.training.tasks import build_regression_loaders, build_classification_loaders
from src.training.task_handler import create_task_handler
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_or_create_pretrained_backbone(config: ProjectConfig, dataset_name: str):
    """加载或创建预训练backbone（参考finetune_pipeline.py）"""
    from src.models.bert.model import create_bert_mlm
    
    # 为了测试，创建一个新的backbone
    udi = UnifiedDataInterface(config=config, dataset=dataset_name)
    vocab_manager = udi.get_vocab(method=config.serialization.method)
    
    # 创建MLM模型作为backbone
    backbone = create_bert_mlm(
        vocab_manager=vocab_manager,
        hidden_size=config.bert.architecture.hidden_size,
        num_hidden_layers=config.bert.architecture.num_hidden_layers,
        num_attention_heads=config.bert.architecture.num_attention_heads,
        intermediate_size=config.bert.architecture.intermediate_size,
        hidden_dropout_prob=config.bert.architecture.hidden_dropout_prob,
        attention_probs_dropout_prob=config.bert.architecture.attention_probs_dropout_prob,
        max_position_embeddings=config.bert.architecture.max_position_embeddings,
        layer_norm_eps=config.bert.architecture.layer_norm_eps,
        type_vocab_size=config.bert.architecture.type_vocab_size,
        initializer_range=config.bert.architecture.initializer_range,
    )
    return backbone


def test_dataset_finetune(dataset_name: str, config: ProjectConfig):
    """测试单个数据集的完整微调流程"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 测试数据集: {dataset_name}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. 为每个数据集设置正确的目标属性
        if dataset_name == "zinc":
            config.task.target_property = "logP_SA_cycle_normalized"
        elif dataset_name == "synthetic":
            config.task.target_property = "label"
        elif dataset_name == "peptides_func":
            config.task.target_property = None  # 多标签分类，使用全部标签
        elif dataset_name == "peptides_struct":
            config.task.target_property = None  # 多目标回归，使用全部目标
        else:
            config.task.target_property = None
        
        # 2. 创建UnifiedDataInterface
        udi = UnifiedDataInterface(config=config, dataset=dataset_name)
        method = config.serialization.method
        
        # 3. 从数据加载器获取任务信息
        from src.data.unified_data_factory import get_dataloader
        loader = get_dataloader(dataset_name, config)
        task_type = loader.get_dataset_task_type()
        num_classes = loader.get_num_classes()
        
        logger.info(f"  任务类型: {task_type}")
        logger.info(f"  类别/目标数: {num_classes}")
        
        # 3. 创建任务处理器
        task_handler = create_task_handler(loader)
        
        # 4. 加载或创建预训练backbone
        pretrained = load_or_create_pretrained_backbone(config, dataset_name)
        
        # 5. 创建统一模型
        model, _ = create_model_from_udi(
            udi=loader,
            pretrained_model=pretrained,
            pooling_method='mean'
        )
        
        logger.info(f"  ✅ 模型创建成功")
        logger.info(f"  输出维度: {task_handler.output_dim}")
        logger.info(f"  主要指标: {task_handler.primary_metric}")
        
        # 6. 构建数据加载器（使用项目标准方法）
        if task_handler.is_regression_task():
            train_dl, val_dl, test_dl, normalizer = build_regression_loaders(
                config, pretrained, udi, method
            )
            logger.info("  使用回归数据加载器")
        else:
            train_dl, val_dl, test_dl = build_classification_loaders(
                config, pretrained, udi, method
            )
            normalizer = None
            logger.info("  使用分类数据加载器")
        
        logger.info(f"  训练批次数: {len(train_dl)}")
        logger.info(f"  批次大小: {config.bert.finetuning.batch_size}")
        
        # 7. 测试前向传播
        logger.info("\n🔄 测试前向传播...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # 获取一个批次的数据
        for batch in train_dl:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logger.info(f"  输入形状: {input_ids.shape}")
            logger.info(f"  注意力掩码形状: {attention_mask.shape}")
            logger.info(f"  标签形状: {labels.shape}")
            
            # 测试模型输出
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                logger.info(f"  模型输出形状: {outputs['outputs'].shape}")
                
                # 计算损失
                loss = task_handler.compute_loss(outputs['outputs'], labels)
                logger.info(f"  损失值: {loss.item():.4f}")
                
                # 获取预测
                predictions = task_handler.get_predictions(outputs['outputs'])
                logger.info(f"  预测形状: {predictions.shape}")
                
                # 对于分类任务，获取概率
                if task_handler.is_classification_task():
                    probs = task_handler.get_probabilities(outputs['outputs'])
                    logger.info(f"  概率形状: {probs.shape}")
            
            break  # 只测试一个批次
        
        logger.info(f"\n✅ {dataset_name} 测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ {dataset_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始测试统一模型微调流程...")
    
    config = ProjectConfig()
    
    # 测试不同类型的数据集
    test_datasets = [
        "zinc",           # 回归任务
        "synthetic",      # 分类任务
        "peptides_func",  # 多标签分类
        "peptides_struct" # 多目标回归
    ]
    
    results = {}
    for dataset in test_datasets:
        results[dataset] = test_dataset_finetune(dataset, config)
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("📊 测试结果汇总:")
    logger.info(f"{'='*60}")
    
    for dataset, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"  {dataset}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 所有测试通过! 统一模型系统可以处理各种任务类型。")
    else:
        logger.info("\n⚠️ 部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
