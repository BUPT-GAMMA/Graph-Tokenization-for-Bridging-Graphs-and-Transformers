"""
统一评估脚本
===========

评估所有训练好的模型并生成对比报告
"""

import torch
import json
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, List

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, VAL_RATIO, SEED,
    MLP_CONFIG, LENET_CONFIG, BERT_CONFIG,
    GRAYSCALE_VOCAB_SIZE, NUM_CLASSES,
    CHECKPOINTS_DIR, RESULTS_DIR, DEVICE,
    get_checkpoint_path, get_result_path, get_bpe_model_path
)
from data import get_mnist_dataloaders
from models import MLPClassifier, LeNet5, TransformerClassifier
from training_utils import evaluate
from data.bpe_processor import ImageBPEProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model_and_evaluate(
    model_name: str,
    device: str
) -> Dict[str, Any]:
    """
    加载模型并评估
    
    Returns:
        评估结果字典
    """
    logger.info(f"\n评估模型: {model_name}")
    logger.info("-" * 50)
    
    # 加载结果文件（如果存在）
    result_path = get_result_path(model_name)
    if not result_path.exists():
        logger.warning(f"  结果文件不存在: {result_path}")
        return None
    
    with open(result_path, 'r') as f:
        results = json.load(f)
    
    # 检查检查点是否存在
    checkpoint_path = get_checkpoint_path(model_name)
    if not checkpoint_path.exists():
        logger.warning(f"  检查点不存在: {checkpoint_path}")
        return results
    
    # 根据模型类型创建模型
    criterion = torch.nn.CrossEntropyLoss()
    
    if model_name == "mlp":
        model = MLPClassifier(**MLP_CONFIG)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估
        _, _, test_loader = get_mnist_dataloaders(
            str(DATA_DIR), BATCH_SIZE, NUM_WORKERS, VAL_RATIO, "flatten", SEED
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
    elif model_name == "lenet":
        model = LeNet5(**LENET_CONFIG)
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估
        _, _, test_loader = get_mnist_dataloaders(
            str(DATA_DIR), BATCH_SIZE, NUM_WORKERS, VAL_RATIO, "image", SEED
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
    elif "transformer" in model_name:
        # Transformer系列模型
        is_bpe = "bpe" in model_name
        transformer_type = model_name.split("_")[-1]  # "bert" or "gte"
        
        if is_bpe:
            # BPE+Transformer需要特殊处理
            # 这里简化处理，直接使用保存的结果
            logger.info(f"  使用保存的结果（BPE模型评估较复杂）")
            return results
        else:
            # 普通Transformer
            model = TransformerClassifier(
                vocab_size=GRAYSCALE_VOCAB_SIZE,
                num_classes=NUM_CLASSES,
                transformer_config=BERT_CONFIG,
                transformer_type=transformer_type,
                pooling_method=BERT_CONFIG.get('pooling_method', 'cls')
            )
            model = model.to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 评估
            _, _, test_loader = get_mnist_dataloaders(
                str(DATA_DIR), BATCH_SIZE, NUM_WORKERS, VAL_RATIO, "sequence", SEED
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    else:
        logger.warning(f"  未知的模型类型: {model_name}")
        return results
    
    logger.info(f"  测试集 - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    logger.info(f"  (保存的结果: {results['final_test_acc']:.4f})")
    
    # 更新结果
    results['evaluated_test_acc'] = test_acc
    results['evaluated_test_loss'] = test_loss
    
    return results


def main(args):
    """主流程"""
    
    logger.info("="*60)
    logger.info("统一模型评估")
    logger.info("="*60)
    
    # 所有模型名称
    model_names = [
        "mlp",
        "lenet",
        "transformer_bert",
        "transformer_gte",
        "bpe_transformer_bert",
        "bpe_transformer_gte"
    ]
    
    # 评估所有模型
    all_results = {}
    for model_name in model_names:
        if args.models and model_name not in args.models:
            continue
        
        results = load_model_and_evaluate(model_name, args.device)
        if results is not None:
            all_results[model_name] = results
    
    # 汇总结果
    logger.info("\n" + "="*60)
    logger.info("评估汇总")
    logger.info("="*60)
    
    summary = []
    for model_name, results in all_results.items():
        summary.append({
            'model': model_name,
            'test_acc': results.get('final_test_acc', 0.0),
            'val_acc': results.get('best_val_acc', 0.0),
            'params': results.get('total_params', 0),
            'time': results.get('training_time_total', 0.0)
        })
    
    # 按测试准确率排序
    summary.sort(key=lambda x: x['test_acc'], reverse=True)
    
    logger.info(f"\n{'模型':<25} {'测试准确率':>12} {'验证准确率':>12} {'参数量':>12} {'训练时间(s)':>12}")
    logger.info("-" * 80)
    for item in summary:
        logger.info(f"{item['model']:<25} {item['test_acc']:>12.4f} "
                   f"{item['val_acc']:>12.4f} {item['params']:>12,} "
                   f"{item['time']:>12.2f}")
    
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一评估所有模型")
    parser.add_argument("--models", nargs="+", default=None,
                       help="指定要评估的模型（默认评估所有）")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)

