"""
训练工具函数
============

提供通用的训练循环、评估和结果记录功能
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingMetrics:
    """训练指标记录器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epoch_times = []
        self.learning_rates = []
    
    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time: float,
        lr: float
    ):
        """记录一个epoch的指标"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.epoch_times.append(epoch_time)
        self.learning_rates.append(lr)
    
    def to_dict(self) -> Dict[str, List]:
        """转换为字典"""
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'epoch_times': self.epoch_times,
            'learning_rates': self.learning_rates
        }
    
    def get_best_val_acc(self) -> float:
        """获取最佳验证准确率"""
        return max(self.val_accs) if self.val_accs else 0.0
    
    def get_best_epoch(self) -> int:
        """获取最佳验证准确率对应的epoch"""
        if not self.val_accs:
            return 0
        best_idx = self.val_accs.index(max(self.val_accs))
        return self.epochs[best_idx]


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    log_interval: int = 50,
    preprocess_fn: Optional[Callable] = None
) -> tuple[float, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        log_interval: 日志打印间隔
        preprocess_fn: 数据预处理函数（可选）
    
    Returns:
        (avg_loss, avg_acc): 平均损失和准确率
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 数据预处理（如果需要）
        if preprocess_fn is not None:
            data = preprocess_fn(data)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)
        
        # 打印日志
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            logger.info(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                       f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    preprocess_fn: Optional[Callable] = None
) -> tuple[float, float]:
    """
    评估模型
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        preprocess_fn: 数据预处理函数（可选）
    
    Returns:
        (avg_loss, avg_acc): 平均损失和准确率
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # 数据预处理（如果需要）
            if preprocess_fn is not None:
                data = preprocess_fn(data)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    num_epochs: int,
    checkpoint_dir: Path,
    model_name: str,
    scheduler: Optional[_LRScheduler] = None,
    log_interval: int = 50,
    save_every_epoch: bool = False,
    save_best_only: bool = True,
    preprocess_fn: Optional[Callable] = None
) -> TrainingMetrics:
    """
    完整的训练流程
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        checkpoint_dir: 检查点保存目录
        model_name: 模型名称
        scheduler: 学习率调度器（可选）
        log_interval: 日志打印间隔
        save_every_epoch: 是否每个epoch保存
        save_best_only: 是否只保存最佳模型
        preprocess_fn: 数据预处理函数（可选）
    
    Returns:
        训练指标
    """
    metrics = TrainingMetrics()
    best_val_acc = 0.0
    
    logger.info(f"开始训练模型: {model_name}")
    logger.info(f"  总epoch数: {num_epochs}")
    logger.info(f"  训练样本: {len(train_loader.dataset)}")
    logger.info(f"  验证样本: {len(val_loader.dataset)}")
    
    total_train_time = 0.0
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        logger.info(f"\nEpoch {epoch}/{num_epochs}")
        logger.info("-" * 50)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            log_interval, preprocess_fn
        )
        
        # 验证
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, preprocess_fn
        )
        
        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        # 记录指标
        metrics.add_epoch(
            epoch, train_loss, train_acc, val_loss, val_acc,
            epoch_time, current_lr
        )
        
        # 打印摘要
        logger.info(f"训练 - Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logger.info(f"验证 - Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        logger.info(f"时间: {epoch_time:.2f}s LR: {current_lr:.6f}")
        
        # 保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_best_only or save_every_epoch:
                best_path = checkpoint_dir / f"{model_name}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, best_path)
                logger.info(f"保存最佳模型到: {best_path}")
        
        if save_every_epoch and not save_best_only:
            epoch_path = checkpoint_dir / f"{model_name}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, epoch_path)
    
    logger.info(f"\n训练完成！")
    logger.info(f"  总时间: {total_train_time:.2f}s")
    logger.info(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {metrics.get_best_epoch()})")
    
    return metrics


def save_training_results(
    model_name: str,
    config: Dict[str, Any],
    metrics: TrainingMetrics,
    test_acc: float,
    total_params: int,
    total_time: float,
    save_path: Path
):
    """
    保存训练结果到JSON文件
    
    Args:
        model_name: 模型名称
        config: 模型配置
        metrics: 训练指标
        test_acc: 测试集准确率
        total_params: 模型参数量
        total_time: 总训练时间
        save_path: 保存路径
    """
    results = {
        'model_name': model_name,
        'config': config,
        'training_history': metrics.to_dict(),
        'best_val_acc': metrics.get_best_val_acc(),
        'best_epoch': metrics.get_best_epoch(),
        'final_test_acc': test_acc,
        'total_params': total_params,
        'training_time_total': total_time
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练结果已保存到: {save_path}")


def load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: str
) -> nn.Module:
    """
    加载最佳检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        加载权重后的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"加载检查点: {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    return model

