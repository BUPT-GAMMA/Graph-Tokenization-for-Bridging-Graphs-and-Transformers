"""
通用训练/评估循环（精简版）
=========================

仅保留最小必要逻辑；日志与可视化由上层注入（WandB 或控制台）。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable, Literal
import time
from tqdm import tqdm
import torch
from src.utils.logger import get_logger

logger = get_logger('tokenizerGraph.training.loops')
logger.propagate = False


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    max_grad_norm: float,
    task_handler,  # 🆕 必需参数，统一架构要求
    on_step: Optional[Callable[[int, float, Optional[float]], None]] = None,
    log_interval: int = 100,
    epoch_num: int = 1,
    total_epochs: int = 1,
    log_style: Literal["online", "offline"] = "online",
    config = None,  # 🆕 用于一致性正则化配置
) -> Dict[str, Any]:
    model.train()
    epoch_loss = 0.0
    steps = 0
    start = time.time()

    progress_desc = f"Epoch {epoch_num}/{total_epochs} - Training"
    steps_per_epoch = len(dataloader)
    pbar = tqdm(dataloader, desc=progress_desc) if log_style == "online" else None
    next_percent_checkpoint = 10 if log_style == "offline" else None
    
    # 为特征混合准备数据
    dataloader_iter = iter(dataloader)
    
    for batch in dataloader:
        # if steps==10: break;
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        # 🆕 统一架构：所有模型都使用TaskHandler计算损失
        if task_handler is None:
            raise ValueError("统一架构要求提供task_handler参数")
            
        # 创建增强器（简洁的方式）
        from src.training.augmentation import create_augmentation
        augmentation = create_augmentation(config)
        
        # 特征混合增强（仅用于回归任务）
        if (augmentation and augmentation.should_use_feature_mixup() and 
            task_handler.is_regression_task()):
            try:
                # 尝试获取下一个batch用于混合
                next_batch = next(dataloader_iter)
                next_input_ids = next_batch['input_ids'].to(device)
                next_attention_mask = next_batch['attention_mask'].to(device) 
                next_labels = next_batch['labels'].to(device)
                
                # 前向传播获取特征
                outputs1 = model(input_ids, attention_mask)
                outputs2 = model(next_input_ids, next_attention_mask)
                
                # 在特征空间混合
                if 'pooled' in outputs1 and 'pooled' in outputs2:
                    mixed_batch, lam = augmentation.prepare_feature_mixup_batch(
                        {'labels': labels}, {'labels': next_labels}
                    )
                    
                    if lam > 0:
                        # 混合特征
                        mixed_features = augmentation.mix_features(
                            outputs1['pooled'], outputs2['pooled'], lam
                        )
                        # 混合标签
                        mixed_labels = augmentation.mix_labels(
                            labels, next_labels, lam, task_handler.task_type
                        )
                        
                        # 计算混合后的预测
                        mixed_outputs = model.task_head(mixed_features)
                        loss = task_handler.compute_loss(mixed_outputs, mixed_labels)
                    else:
                        loss = task_handler.compute_loss(outputs1['outputs'], labels)
                else:
                    loss = task_handler.compute_loss(outputs1['outputs'], labels)
                    
            except StopIteration:
                # 没有更多batch，使用标准训练
                outputs = model(input_ids, attention_mask) 
                loss = task_handler.compute_loss(outputs['outputs'], labels)
                
        elif augmentation and augmentation.should_use_consistency_regularization():
            # R-Drop：两次前向传播
            outputs1 = model(input_ids, attention_mask)
            outputs2 = model(input_ids, attention_mask)
            
            total_loss, task_loss, consistency_loss = task_handler.compute_loss_with_consistency(
                outputs1['outputs'], outputs2['outputs'], labels, 
                augmentation.aug_config.consistency_alpha
            )
            loss = total_loss
        else:
            # 标准训练
            outputs = model(input_ids, attention_mask)
            
            # 简洁的高斯噪声增强：在输出特征上添加噪声
            if (augmentation and augmentation.should_use_gaussian_noise() and 
                model.task_type != 'mlm' and 'pooled' in outputs):
                outputs['pooled'] = augmentation.apply_gaussian_noise(outputs['pooled'])
                # 重新计算任务输出
                outputs['outputs'] = model.task_head(outputs['pooled'])
                
            loss = task_handler.compute_loss(outputs['outputs'], labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        steps += 1

        # 日志输出：online 使用 tqdm；offline 每完成10%输出一次摘要
        if log_style == "online":
            if steps % 20 == 0 or steps == 1:
                avg_loss = epoch_loss / steps
                if pbar is not None:
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    pbar.update(20 if steps != 1 else 1)
        else:
            if steps_per_epoch:
                progress_pct = int(steps * 100 / max(1, steps_per_epoch))
                if next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                    avg_loss = epoch_loss / steps
                    elapsed = time.time() - start
                    est_total = (elapsed / max(1, steps)) * steps_per_epoch
                    eta = max(0.0, est_total - elapsed)
                    logger.info(f"[Offline] {progress_desc} {progress_pct}% | loss={avg_loss:.4f} | elapsed={elapsed:.1f}s | eta={eta:.1f}s")
                    while next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                        next_percent_checkpoint += 10

        if on_step is not None and (steps % max(1, log_interval) == 0):
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
            on_step(steps, loss.item(), current_lr)
    # 补全进度条到总数（online）
    if log_style == "online" and pbar is not None:
        if steps % 20 != 0:
            pbar.update(steps % 20)
            avg_loss = epoch_loss / steps
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        pbar.close()

    return {
        'loss': epoch_loss / max(steps, 1),
        'time': time.time() - start,
        'steps': steps,
    }


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, task_handler, epoch_num: int = 1, desc: str = "Validation", log_style: Literal["online", "offline"] = "online") -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    steps = 0

    progress_desc = f"Epoch {epoch_num} - {desc}"
    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    pbar = tqdm(dataloader, desc=progress_desc) if log_style == "online" else None
    next_percent_checkpoint = 10 if log_style == "offline" else None
    start = time.time()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 🆕 统一架构：所有模型都使用TaskHandler计算损失
        outputs = model(input_ids, attention_mask)
        loss = task_handler.compute_loss(outputs['outputs'], labels)
            
        total_loss += loss.item()
        steps += 1

        if log_style == "online":
            if steps % 20 == 0 or steps == 1:
                avg_loss = total_loss / steps
                if pbar is not None:
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    pbar.update(20 if steps != 1 else 1)
        else:
            if steps_per_epoch:
                progress_pct = int(steps * 100 / max(1, steps_per_epoch))
                if next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                    avg_loss = total_loss / steps
                    elapsed = time.time() - start
                    est_total = (elapsed / max(1, steps)) * steps_per_epoch
                    eta = max(0.0, est_total - elapsed)
                    logger.info(f"[Offline] {progress_desc} {progress_pct}% | loss={avg_loss:.4f} | elapsed={elapsed:.1f}s | eta={eta:.1f}s")
                    while next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                        next_percent_checkpoint += 10

    if log_style == "online" and pbar is not None:
        if steps % 20 != 0:
            pbar.update(steps % 20)
            avg_loss = total_loss / max(steps, 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        pbar.close()

    return {
        'loss': total_loss / max(steps, 1),
        'steps': steps,
    }


