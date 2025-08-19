#!/usr/bin/env python3
"""
测试统一模型流水线
==================

使用真实数据集测试统一模型系统的完整流程。
"""

import torch
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader
from src.models.bert.heads import create_model_from_udi
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_real_dataset_pipeline(dataset_name: str, config: ProjectConfig):
    """测试真实数据集的完整流程"""
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 测试数据集: {dataset_name}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. 创建数据加载器
        logger.info("📂 加载数据...")
        udi = get_dataloader(dataset_name, config)
        task_type = udi.get_dataset_task_type()
        num_classes = udi.get_num_classes()
        
        # 加载数据
        train_data, val_data, test_data, train_labels, val_labels, test_labels = udi.load_data()
        
        logger.info(f"  任务类型: {task_type}")
        logger.info(f"  类别/目标数: {num_classes}")
        logger.info(f"  训练集大小: {len(train_data)}")
        
        # 2. 获取序列化方法
        method = config.serialization.method
        logger.info(f"  序列化方法: {method}")
        
        # 3. 获取一些示例数据（需要使用UnifiedDataInterface）
        from src.data.unified_data_interface import UnifiedDataInterface
        data_interface = UnifiedDataInterface(config=config, dataset=dataset_name)
        data_interface._loader = udi  # 复用已加载的loader
        
        # 获取序列化后的序列
        (train_seqs_with_id, train_props), _, _ = data_interface.get_training_data(method)
        train_seqs = [seq for _, seq in train_seqs_with_id[:5]]
        train_labels_sample = train_labels[:5]
        
        logger.info(f"  示例序列长度: {[len(seq) for seq in train_seqs]}")
        logger.info(f"  示例标签: {train_labels_sample}")
        
        # 4. 创建预训练模型（使用mock）
        logger.info("\n🔧 创建预训练模型...")
        logger.info("  使用随机初始化的模型进行测试...")
        
        # 创建一个mock预训练模型
        from src.models.bert import BertConfig, VocabManager
        
        # 从UDI获取词表（这是项目的标准做法）
        vocab_manager = data_interface.get_vocab(method=method)
        
        # 计算有效的最大序列长度（参考pretrain_pipeline.py）
        from src.models.bert.data import compute_effective_max_length
        # 获取所有序列来计算最大长度
        (all_train_seqs, _), (all_val_seqs, _), (all_test_seqs, _) = data_interface.get_training_data(method)
        all_seqs_for_length = ([seq for _, seq in all_train_seqs[:1000]] + 
                               [seq for _, seq in all_val_seqs[:100]] + 
                               [seq for _, seq in all_test_seqs[:100]])
        effective_max_length = compute_effective_max_length(all_seqs_for_length, config)
        logger.info(f"  有效最大序列长度: {effective_max_length}")
        
        # 创建配置
        bert_config = BertConfig(
            vocab_size=vocab_manager.vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout=0.1,
            max_position_embeddings=effective_max_length,  # 使用计算出的有效长度
            layer_norm_eps=1e-12,
            pad_token_id=vocab_manager.pad_token_id
        )
        
        # 创建mock预训练模型
        class MockPretrained:
            def __init__(self, config, vocab_manager):
                self.config = config
                self.vocab_manager = vocab_manager
                # 创建一个BERT模型实例
                from transformers import BertModel
                hf_config = config.to_hf_config()
                self.bert = BertModel(hf_config)
        
        pretrained = MockPretrained(bert_config, vocab_manager)
        
        # 5. 创建统一模型和任务处理器
        logger.info("\n📦 创建统一模型...")
        model, task_handler = create_model_from_udi(
            udi=udi,
            pretrained_model=pretrained,
            pooling_method='mean'  # 使用默认的mean池化
        )
        
        logger.info(f"  ✅ 模型创建成功")
        logger.info(f"  输出维度: {task_handler.output_dim}")
        logger.info(f"  主要指标: {task_handler.primary_metric}")
        logger.info(f"  是否最大化指标: {task_handler.should_maximize_metric}")
        
        # 6. 测试前向传播
        logger.info("\n🔄 测试前向传播...")
        
        # 创建一个小批次
        batch_size = min(4, len(train_seqs))
        # 确保max_len不超过模型的最大位置编码长度
        max_len = min(
            max(len(seq) for seq in train_seqs[:batch_size]),
            effective_max_length
        )
        
        # 填充或截断序列
        padded_seqs = []
        attention_masks = []
        for seq in train_seqs[:batch_size]:
            # 如果序列太长，截断它
            if len(seq) > max_len:
                seq = seq[:max_len]
            # 填充序列
            padded_seq = seq + [pretrained.vocab_manager.pad_token_id] * (max_len - len(seq))
            padded_seqs.append(padded_seq)
            attention_masks.append([1] * len(seq) + [0] * (max_len - len(seq)))
        
        input_ids = torch.tensor(padded_seqs)
        attention_mask = torch.tensor(attention_masks)
        
        # 准备标签
        batch_labels = train_labels_sample[:batch_size]
        if task_type == "regression":
            labels = torch.tensor(batch_labels).float().unsqueeze(-1)
        elif task_type == "multi_target_regression":
            labels = torch.tensor(batch_labels).float()
        elif task_type in ["binary_classification", "classification"]:
            labels = torch.tensor(batch_labels).long()
        elif task_type == "multi_label_classification":
            labels = torch.tensor(batch_labels).float()
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logger.info(f"  输出形状: {outputs['outputs'].shape}")
            
            # 计算损失
            loss = task_handler.compute_loss(outputs['outputs'], labels)
            logger.info(f"  损失值: {loss.item():.4f}")
            
            # 获取预测
            predictions = task_handler.get_predictions(outputs['outputs'])
            logger.info(f"  预测值: {predictions}")
            
            # 获取概率（仅分类）
            probs = task_handler.get_probabilities(outputs['outputs'])
            if probs is not None:
                logger.info(f"  概率值: {probs}")
        
        logger.info(f"\n✅ {dataset_name} 测试成功!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ {dataset_name} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始测试统一模型流水线...")
    
    config = ProjectConfig()
    
    # 测试不同类型的数据集
    test_cases = [
        ("zinc", "回归任务"),
        ("synthetic", "分类任务"),
        ("peptides_func", "多标签分类"),
        ("peptides_struct", "多目标回归"),
    ]
    
    results = {}
    
    for dataset, desc in test_cases:
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 {desc}: {dataset}")
        logger.info(f"{'='*60}")
        
        results[dataset] = test_real_dataset_pipeline(dataset, config)
    
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
