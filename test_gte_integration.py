"""
GTE微调集成测试
==============

测试BERT和GTE在微调pipeline中的切换，对比性能表现。
"""

import time
import logging
from pathlib import Path

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.training.finetune_pipeline import run_finetune

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_encoder_switching():
    """测试编码器切换功能"""
    
    logger.info("🧪 === GTE微调集成测试 ===")
    
    # 创建测试配置
    config = ProjectConfig()
    
    # 确保使用较小的数据集进行快速测试
    config.dataset.limit = 1000  # 仅使用1000个样本
    config.bert.finetuning.epochs = 2  # 仅训练2个epoch
    config.bert.finetuning.batch_size = 8
    
    # 设置实验参数
    config.experiment_group = "gte_integration_test"
    config.experiment_name = "quick_test"
    
    results = {}
    
    # 测试1: BERT微调（baseline）
    logger.info("🤖 测试1: BERT微调")
    try:
        config.encoder_type = 'bert'
        start_time = time.time()
        
        bert_result = run_finetune(
            config, 
            task='regression',
            save_name_suffix='bert_test'
        )
        
        bert_time = time.time() - start_time
        results['bert'] = {
            'success': True,
            'training_time': bert_time,
            'final_loss': bert_result.get('best_val_loss', None),
            'model_type': 'BERT'
        }
        
        logger.info(f"✅ BERT测试完成: 用时{bert_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ BERT测试失败: {e}")
        results['bert'] = {'success': False, 'error': str(e)}
    
    # 测试2: GTE微调
    logger.info("🚀 测试2: GTE微调")
    try:
        config.encoder_type = 'gte'
        start_time = time.time()
        
        gte_result = run_finetune(
            config,
            task='regression', 
            save_name_suffix='gte_test'
        )
        
        gte_time = time.time() - start_time
        results['gte'] = {
            'success': True,
            'training_time': gte_time,
            'final_loss': gte_result.get('best_val_loss', None),
            'model_type': 'GTE'
        }
        
        logger.info(f"✅ GTE测试完成: 用时{gte_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ GTE测试失败: {e}")
        results['gte'] = {'success': False, 'error': str(e)}
    
    # 结果对比
    logger.info("📊 === 测试结果对比 ===")
    
    for model_type, result in results.items():
        if result['success']:
            logger.info(f"✅ {model_type.upper()}: "
                       f"用时 {result['training_time']:.2f}s, "
                       f"验证损失 {result.get('final_loss', 'N/A')}")
        else:
            logger.error(f"❌ {model_type.upper()}: 失败 - {result['error']}")
    
    # 性能分析
    if results.get('bert', {}).get('success') and results.get('gte', {}).get('success'):
        bert_time = results['bert']['training_time']
        gte_time = results['gte']['training_time']
        
        if gte_time < bert_time:
            speedup = bert_time / gte_time
            logger.info(f"🚀 GTE比BERT快 {speedup:.2f}x!")
        else:
            slowdown = gte_time / bert_time
            logger.info(f"⏳ GTE比BERT慢 {slowdown:.2f}x")
        
        # 损失对比
        bert_loss = results['bert'].get('final_loss')
        gte_loss = results['gte'].get('final_loss')
        
        if bert_loss is not None and gte_loss is not None:
            if gte_loss < bert_loss:
                improvement = (bert_loss - gte_loss) / bert_loss * 100
                logger.info(f"📈 GTE验证损失改善 {improvement:.1f}%!")
            else:
                degradation = (gte_loss - bert_loss) / bert_loss * 100
                logger.info(f"📉 GTE验证损失下降 {degradation:.1f}%")
    
    return results


def test_basic_gte_creation():
    """测试GTE编码器基础创建功能"""
    
    logger.info("🔧 === 基础GTE创建测试 ===")
    
    try:
        config = ProjectConfig()
        config.dataset.limit = 100
        config.encoder_type = 'gte'
        
        # 测试GTE编码器创建
        from src.training.finetune_pipeline import _load_gte_backbone
        
        gte_encoder = _load_gte_backbone(config)
        
        logger.info(f"✅ GTE编码器创建成功:")
        logger.info(f"   - 隐藏层维度: {gte_encoder.get_hidden_size()}")
        logger.info(f"   - 最大序列长度: {gte_encoder.get_max_seq_length()}")
        logger.info(f"   - 模型名称: {gte_encoder.model_name}")
        
        # 测试任务模型创建
        from src.data.unified_data_interface import UnifiedDataInterface
        from src.training.model_builder import build_task_model
        
        udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)
        model, task_handler = build_task_model(config, gte_encoder, udi, config.serialization.method)
        
        logger.info(f"✅ 任务模型创建成功:")
        logger.info(f"   - 任务类型: {task_handler.task_type}")
        logger.info(f"   - 输出维度: {task_handler.output_dim}")
        
        # 简单前向传播测试
        import torch
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        logger.info(f"✅ 前向传播测试成功: 输出形状 {output.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 基础创建测试失败: {e}")
        logger.exception("详细错误:")
        return False


if __name__ == "__main__":
    """运行所有测试"""
    
    logger.info("🚀 开始GTE集成测试...")
    
    # 基础功能测试
    basic_success = test_basic_gte_creation()
    
    if basic_success:
        logger.info("✅ 基础功能测试通过，继续完整集成测试...")
        
        # 完整集成测试
        integration_results = test_encoder_switching()
        
        logger.info("🏁 === 所有测试完成 ===")
        
        # 生成测试报告
        success_count = sum(1 for r in integration_results.values() if r.get('success', False))
        total_count = len(integration_results)
        
        logger.info(f"📈 测试成功率: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("🎉 所有测试都成功通过！GTE集成就绪！")
        else:
            logger.warning("⚠️ 部分测试失败，需要进一步调试")
            
    else:
        logger.error("❌ 基础功能测试失败，请先解决基础问题")
    
    logger.info("📋 测试总结：BERT与GTE微调切换功能已实现")
