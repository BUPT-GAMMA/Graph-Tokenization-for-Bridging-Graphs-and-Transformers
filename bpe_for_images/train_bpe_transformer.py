"""
训练BPE+Transformer分类器
========================

使用BPE压缩后的序列训练Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, Subset
import sys
from pathlib import Path
import argparse
import numpy as np

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, VAL_RATIO, SEED,
    EPOCHS, LR_TRANSFORMER, WEIGHT_DECAY, ADAM_BETAS,
    USE_LR_SCHEDULER, LR_WARMUP_EPOCHS,
    CHECKPOINTS_DIR, RESULTS_DIR,
    BERT_CONFIG, NUM_CLASSES,
    DEVICE, LOG_INTERVAL,
    SAVE_EVERY_EPOCH, SAVE_BEST_ONLY,
    get_checkpoint_path, get_result_path, get_bpe_model_path
)
from data import get_mnist_raw_data, prepare_flattened_sequences
from data.bpe_processor import ImageBPEProcessor
from models import TransformerClassifier
from training_utils import (
    train_model, evaluate, save_training_results,
    load_best_checkpoint
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BPEEncodedMNISTDataset(Dataset):
    """BPE编码的MNIST数据集"""
    
    def __init__(self, encoded_sequences, labels):
        """
        Args:
            encoded_sequences: BPE编码后的序列列表
            labels: 标签列表
        """
        self.encoded_sequences = encoded_sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.encoded_sequences[idx], dtype=torch.long)
        label = int(self.labels[idx])
        return sequence, label


def collate_fn(batch):
    """
    批次整理函数，处理变长序列
    """
    sequences, labels = zip(*batch)
    
    # 找到最大长度
    max_len = max(len(seq) for seq in sequences)
    
    # Padding
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        seq_len = len(seq)
        # Padding到最大长度（使用0作为pad token id）
        padded_seq = torch.cat([
            seq,
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        # 创建attention mask（1表示真实token，0表示padding）
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(max_len - seq_len, dtype=torch.long)
        ])
        padded_sequences.append(padded_seq)
        attention_masks.append(mask)
    
    sequences_tensor = torch.stack(padded_sequences)
    masks_tensor = torch.stack(attention_masks)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return sequences_tensor, labels_tensor


def main(args):
    """主训练流程"""
    
    # 设置随机种子
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    logger.info("="*60)
    logger.info(f"训练BPE+Transformer分类器 (type={args.transformer_type})")
    logger.info("="*60)
    
    # 1. 加载BPE模型
    logger.info("\n1. 加载BPE模型...")
    bpe_model_path = get_bpe_model_path() if args.bpe_model is None else args.bpe_model
    
    if not Path(bpe_model_path).exists():
        logger.error(f"BPE模型不存在: {bpe_model_path}")
        logger.error("请先运行 train_bpe.py 训练BPE模型")
        return
    
    processor = ImageBPEProcessor.load(str(bpe_model_path))
    vocab_size = processor.get_vocab_size()
    logger.info(f"  BPE词汇表大小: {vocab_size}")
    
    # 2. 加载并编码MNIST数据
    logger.info("\n2. 加载和编码MNIST数据...")
    
    # 训练集
    logger.info("  处理训练集...")
    train_images, train_labels = get_mnist_raw_data(str(DATA_DIR), train=True)
    train_sequences = prepare_flattened_sequences(train_images)
    train_encoded, train_stats = processor.encode(train_sequences)
    logger.info(f"    训练集编码完成，平均长度: {train_stats['avg_length']:.1f}")
    
    # 测试集
    logger.info("  处理测试集...")
    test_images, test_labels = get_mnist_raw_data(str(DATA_DIR), train=False)
    test_sequences = prepare_flattened_sequences(test_images)
    test_encoded, test_stats = processor.encode(test_sequences)
    logger.info(f"    测试集编码完成，平均长度: {test_stats['avg_length']:.1f}")
    
    # 3. 创建数据集和数据加载器
    logger.info("\n3. 创建数据加载器...")
    train_dataset = BPEEncodedMNISTDataset(train_encoded, train_labels)
    test_dataset = BPEEncodedMNISTDataset(test_encoded, test_labels)
    
    # 划分训练/验证集
    train_size = len(train_dataset)
    val_size = int(train_size * VAL_RATIO)
    train_size = train_size - val_size
    
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    
    train_subset = Subset(train_dataset, indices[:train_size])
    val_subset = Subset(train_dataset, indices[train_size:])
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    logger.info(f"  数据集划分: 训练={train_size}, 验证={val_size}, 测试={len(test_dataset)}")
    
    # 4. 创建模型
    logger.info("\n4. 创建模型...")
    model = TransformerClassifier(
        vocab_size=vocab_size,  # BPE扩展后的词汇表
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
    
    # 5. 设置损失函数和优化器
    logger.info("\n5. 配置训练...")
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
    
    # 6. 训练模型
    logger.info("\n6. 开始训练...")
    import time
    train_start = time.time()
    
    model_name = f"bpe_transformer_{args.transformer_type}"
    
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
    
    # 7. 加载最佳模型并在测试集上评估
    logger.info("\n7. 测试集评估...")
    best_checkpoint = get_checkpoint_path(model_name)
    model = load_best_checkpoint(model, best_checkpoint, args.device)
    
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, args.device
    )
    
    logger.info(f"  测试集 - Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
    
    # 8. 保存结果
    logger.info("\n8. 保存结果...")
    result_config = {
        **BERT_CONFIG,
        'transformer_type': args.transformer_type,
        'vocab_size': vocab_size,
        'num_classes': NUM_CLASSES,
        'bpe_num_merges': processor.num_merges,
        'bpe_min_frequency': processor.min_frequency,
        'avg_sequence_length': train_stats['avg_length']
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
    logger.info(f"BPE+Transformer ({args.transformer_type}) 训练完成！")
    logger.info(f"  最佳验证准确率: {metrics.get_best_val_acc():.4f}")
    logger.info(f"  测试集准确率: {test_acc:.4f}")
    logger.info(f"  总训练时间: {total_time:.2f}s")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练BPE+Transformer分类器")
    parser.add_argument("--transformer_type", type=str, default="bert",
                       choices=["bert", "gte"],
                       help="Transformer类型")
    parser.add_argument("--bpe_model", type=str, default=None,
                       help="BPE模型路径（默认使用config中的路径）")
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

