"""
通用训练/评估循环（精简版）
=========================

仅保留最小必要逻辑；日志与可视化由上层注入（WandB 或控制台）。
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable
import time
from tqdm import tqdm
import torch


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    max_grad_norm: float,
    on_step: Optional[Callable[[int, float, Optional[float]], None]] = None,
    log_interval: int = 100,
    epoch_num: int = 1,
    total_epochs: int = 1,
) -> Dict[str, Any]:
    model.train()
    epoch_loss = 0.0
    steps = 0
    start = time.time()

    # 创建带epoch信息的进度条
    progress_desc = f"Epoch {epoch_num}/{total_epochs} - Training"
    pbar = tqdm(dataloader, desc=progress_desc)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_loss += loss.item()
        steps += 1

        # 每10个batch更新一次进度条显示
        if steps % 20 == 0 or steps == 1:
            avg_loss = epoch_loss / steps
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            pbar.update(20 if steps != 1 else 1)

        if on_step is not None and (steps % max(1, log_interval) == 0):
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
            on_step(steps, loss.item(), current_lr)
    # 补全进度条到总数
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
def evaluate_epoch(model, dataloader, device, epoch_num: int = 1, desc: str = "Validation") -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    steps = 0

    # 创建带epoch信息的进度条
    progress_desc = f"Epoch {epoch_num} - {desc}"
    # pbar = tqdm(dataloader, desc=progress_desc)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        total_loss += loss.item()
        steps += 1

        # 更新进度条显示当前损失
        # avg_loss = total_loss / steps
        # pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return {
        'loss': total_loss / max(steps, 1),
        'steps': steps,
    }


