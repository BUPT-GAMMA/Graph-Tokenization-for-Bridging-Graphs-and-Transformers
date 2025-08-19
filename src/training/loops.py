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
    task_handler,
    on_step: Optional[Callable[[int, float, Optional[float]], None]] = None,
    log_interval: int = 100,
    epoch_num: int = 1,
    total_epochs: int = 1,
    log_style: Literal["online", "offline"] = "online",
) -> Dict[str, Any]:
    model.train()
    epoch_loss = 0.0
    steps = 0
    start = time.time()

    progress_desc = f"Epoch {epoch_num}/{total_epochs} - Training"
    steps_per_epoch = len(dataloader)
    pbar = tqdm(dataloader, desc=progress_desc) if log_style == "online" else None
    next_percent_checkpoint = 10 if log_style == "offline" else None
    for batch in dataloader:
        if steps==10: break;
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
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
def evaluate_epoch(model, dataloader, device, epoch_num: int = 1, desc: str = "Validation", log_style: Literal["online", "offline"] = "online") -> Dict[str, Any]:
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
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
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


