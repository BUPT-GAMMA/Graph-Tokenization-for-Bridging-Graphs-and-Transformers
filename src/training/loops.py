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
    prev_batch = None
    
    for batch in dataloader:
        # if steps==10: break;
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        # 🆕 统一架构：所有模型都使用TaskHandler计算损失
        if task_handler is None:
            raise ValueError("统一架构要求提供task_handler参数")
            
<<<<<<< HEAD
        outputs = model(input_ids, attention_mask)
        loss = task_handler.compute_loss(outputs['outputs'], labels)
=======
        # 创建增强器（简洁的方式）
        from src.training.augmentation import create_augmentation
        # 根据task_handler类型判断任务类型
        if hasattr(task_handler, 'task_type'):
            task_type = task_handler.task_type
        else:
            task_type = "auto"
        augmentation = create_augmentation(config, task_type)
        
        # 计算损失（增强逻辑完全封装在内部）
        current_batch = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        if augmentation:
            loss = augmentation.compute_training_loss(model, current_batch, task_handler, prev_batch)
        else:
            # 标准训练
            outputs = model(input_ids, attention_mask)
            loss = task_handler.compute_loss(outputs['outputs'], labels)
        
        # 检测loss NaN/Inf
        loss_value = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ NaN/Inf loss detected at step {steps+1}: {loss_value}")
            logger.error(f"   input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
            logger.error("   跳过此步骤，继续训练...")
            optimizer.zero_grad()
            prev_batch = current_batch
            continue
            
        if (epoch_num > 1 and loss_value > 20.0) or (epoch_num > 1 and steps>10 and loss_value > (epoch_loss / (steps + 1)) * 10) :
            logger.warning(f"⚠️ Unusually large loss at step {steps+1}: {loss_value:.4f}")
            
>>>>>>> dev
        loss.backward()
        
        # 检测梯度NaN/Inf和异常大梯度
        total_norm = 0.0
        nan_grads = False
        inf_grads = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    nan_grads = True
                if torch.isinf(p.grad).any():
                    inf_grads = True
                if nan_grads or inf_grads:
                    break
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if nan_grads or inf_grads:
            grad_type = "NaN" if nan_grads else "Inf"
            logger.error(f"❌ {grad_type} gradients detected at step {steps+1}，跳过此步骤，继续训练...")
            optimizer.zero_grad()
            prev_batch = current_batch
            continue
        
        if total_norm > 100.0:
            logger.warning(f"⚠️ Large gradient norm at step {steps+1}: {total_norm:.2f} (max_grad_norm={max_grad_norm})")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss_value
        steps += 1
        # torch.cuda.empty_cache()
        # 日志输出：online 使用 tqdm；offline 每完成10%输出一次摘要
        if log_style == "online":
            if steps % 20 == 0 or steps == 1:
                avg_loss = epoch_loss / steps
                # torch.cuda.empty_cache()
                if pbar is not None:
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    pbar.update(20 if steps != 1 else 1)
        else:
            if steps_per_epoch:
                progress_pct = int(steps * 100 / max(1, steps_per_epoch))
                if next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                    avg_loss = epoch_loss / steps
                    # torch.cuda.empty_cache()
                    elapsed = time.time() - start
                    est_total = (elapsed / max(1, steps)) * steps_per_epoch
                    eta = max(0.0, est_total - elapsed)
                    logger.info(f"[Offline] {progress_desc} {progress_pct}% | loss={avg_loss:.4f} | elapsed={elapsed:.1f}s | eta={eta:.1f}s")
                    while next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                        next_percent_checkpoint += 10

        if on_step is not None and (steps % max(1, log_interval) == 0):
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
            on_step(steps, loss.item(), current_lr)
            
        # 更新prev_batch用于下一次可能的mixup
        prev_batch = current_batch
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
<<<<<<< HEAD
            
        total_loss += loss.item()
=======
        
        # 检测validation loss NaN/Inf
        loss_value = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ NaN/Inf validation loss detected at step {steps+1}: {loss_value}")
            logger.error(f"   input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")
            logger.error("   跳过此验证步骤...")
            continue
            
        if loss_value > 1000.0:
            logger.warning(f"⚠️ Unusually large validation loss at step {steps+1}: {loss_value:.4f}")
            
        total_loss += loss_value
>>>>>>> dev
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


