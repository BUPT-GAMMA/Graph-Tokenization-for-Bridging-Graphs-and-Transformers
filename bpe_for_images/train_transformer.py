"""
训练Transformer分类器（灰度值直接作为token）
=========================================

使用展平的灰度值序列（0-255）直接作为token输入Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
from pathlib import Path
import argparse

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, VAL_RATIO, SEED,
    EPOCHS, LR_TRANSFORMER, WEIGHT_DECAY, ADAM_BETAS,
    USE_LR_SCHEDULER, LR_WARMUP_EPOCHS,
    CHECKPOINTS_DIR, RESULTS_DIR,
    BERT_CONFIG, GRAYSCALE_VOCAB_SIZE, NUM_CLASSES,
    DEVICE, LOG_INTERVAL,
    SAVE_EVERY_EPOCH, SAVE_BEST_ONLY,
    get_checkpoint_path, get_result_path
)
from data import get_mnist_dataloaders
from models import TransformerClassifier
from training_utils import (
    train_model, evaluate, save_training_results,
    load_best_checkpoint
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main(args):
    """主训练流程"""
    
    # 设置随机种子
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    logger.info("="*60)
    logger.info(f"训练Transformer分类器 (type={args.transformer_type})")
    logger.info("="*60)
    
    # 1. 加载数据（序列格式）
    logger.info("\n1. 加载数据...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir=str(DATA_DIR),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        format="sequence",  # Transformer使用序列格式（灰度值0-255）
        seed=SEED
    )
    
    # 2. 创建模型
    logger.info("\n2. 创建模型...")
    model = TransformerClassifier(
        vocab_size=GRAYSCALE_VOCAB_SIZE,  # 256 (0-255灰度值)
        num_classes=NUM_CLASSES,
        transformer_config=BERT_CONFIG,
        transformer_type=args.transformer_type,
        pooling_method=BERT_CONFIG.get('pooling_method', 'cls')
    )
    model = model.to(args.device)
    
    total_params = model.count_parameters()
    encoder_params = model.get_encoder_parameters()
    classifier_params = model.get_classifier_parameters()
    
    logger.info(f"  模型参数量:")
    logger.info(f"    - 总参数: {total_params:,}")
    logger.info(f"    - 编码器: {encoder_params:,}")
    logger.info(f"    - 分类头: {classifier_params:,}")
    
    # 3. 设置损失函数和优化器
    logger.info("\n3. 配置训练...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=ADAM_BETAS
    )
    
    # 学习率调度器
    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - LR_WARMUP_EPOCHS,
            eta_min=args.lr * 0.01
        )
    
    # 4. 训练模型
    logger.info("\n4. 开始训练...")
    import time
    train_start = time.time()
    
    model_name = f"transformer_{args.transformer_type}"
    
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        num_epochs=args.epochs,
        checkpoint_dir=CHECKPOINTS_DIR,
        model_name=model_name,
        scheduler=scheduler,
        log_interval=LOG_INTERVAL,
        save_every_epoch=SAVE_EVERY_EPOCH,
        save_best_only=SAVE_BEST_ONLY
    )
    
    total_time = time.time() - train_start
    
    # 5. 加载最佳模型并在测试集上评估
    logger.info("\n5. 测试集评估...")
    best_checkpoint = get_checkpoint_path(model_name)
    model = load_best_checkpoint(model, best_checkpoint, args.device)
    
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, args.device
    )
    
    logger.info(f"  测试集 - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    
    # 6. 保存结果
    logger.info("\n6. 保存结果...")
    result_config = {
        **BERT_CONFIG,
        'transformer_type': args.transformer_type,
        'vocab_size': GRAYSCALE_VOCAB_SIZE,
        'num_classes': NUM_CLASSES
    }
    
    save_training_results(
        model_name=model_name,
        config=result_config,
        metrics=metrics,
        test_acc=test_acc,
        total_params=total_params,
        total_time=total_time,
        save_path=get_result_path(model_name)
    )
    
    logger.info("\n" + "="*60)
    logger.info(f"Transformer ({args.transformer_type}) 训练完成！")
    logger.info(f"  最佳验证准确率: {metrics.get_best_val_acc():.4f}")
    logger.info(f"  测试集准确率: {test_acc:.4f}")
    logger.info(f"  总训练时间: {total_time:.2f}s")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练Transformer分类器")
    parser.add_argument("--transformer_type", type=str, default="bert",
                       choices=["bert", "gte"],
                       help="Transformer类型")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                       help="批次大小")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=LR_TRANSFORMER,
                       help="学习率")
    parser.add_argument("--device", type=str, default=DEVICE,
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    main(args)

