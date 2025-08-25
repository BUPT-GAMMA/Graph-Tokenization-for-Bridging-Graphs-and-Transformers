from __future__ import annotations

from typing import Optional
import math
import torch


def build_optimizer_and_scheduler(
    model,
    *,
    total_steps: int,
    base_lr: float,
    weight_decay: float,
    head_lr_multiplier: Optional[float] = None,

    warmup_steps: Optional[int] = None,
    warmup_ratio: Optional[float] = None,
):
    """
    构建 AdamW + 线性warmup(可选) + 余弦退火；可选任务头学习率倍率。
    返回 (optimizer, scheduler)
    """
    if head_lr_multiplier is not None and head_lr_multiplier != 1.0:
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ['regression_head', 'classification_head', 'task_head']):
                head_params.append(param)
            else:
                backbone_params.append(param)
        param_groups = [
            {'params': backbone_params, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': base_lr * head_lr_multiplier, 'weight_decay': weight_decay},
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': base_lr, 'weight_decay': weight_decay}]

    optimizer = torch.optim.AdamW(param_groups)

    # 计算 warmup 步数：优先 warmup_steps，其次 warmup_ratio（默认 0.1），最后为 0
    effective_warmup_steps = 0
    if isinstance(warmup_steps, int) and warmup_steps > 0:
        effective_warmup_steps = warmup_steps
    else:
        ratio = 0.1 if warmup_ratio is None else float(max(0.0, warmup_ratio))
        if ratio > 0.0 and total_steps > 0:
            effective_warmup_steps = int(total_steps * ratio)
    # 边界保护
    if effective_warmup_steps >= total_steps:
        effective_warmup_steps = max(0, total_steps - 1)

    # Lambda函数保留备用

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if effective_warmup_steps > 0 and current_step < effective_warmup_steps:
            # 线性 warmup：避免除以 0
            return float(current_step + 1) / float(effective_warmup_steps)
        # 余弦退火阶段
        nonlocal_steps = max(1, total_steps - effective_warmup_steps)
        t = max(0, current_step - effective_warmup_steps)
        progress = min(1.0, float(t) / float(nonlocal_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        eta_min_ratio = 0.01  # 备用参数
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    return optimizer, scheduler


def build_from_config(model, config, *, total_steps: int, stage: str):
    """
    使用统一 config 构建 optimizer/scheduler。
    - 预训练：读取 config.bert.pretraining.{learning_rate, weight_decay}
    - 微调：读取 config.bert.finetuning.{learning_rate, weight_decay, use_layered_lr, head_lr_multiplier}
    """
    if stage == "pretrain":
        # 预训练路径（或默认)
        return build_optimizer_and_scheduler(
            model,
            total_steps=total_steps,
            base_lr=float(config.bert.pretraining.learning_rate),
            weight_decay=float(config.bert.pretraining.weight_decay),
            head_lr_multiplier=None,
            warmup_steps=int(getattr(config.bert.pretraining, 'warmup_steps', 0) or 0),
            warmup_ratio=float(getattr(config.bert.pretraining, 'warmup_ratio', 0.1)),
        )
    elif stage == "finetune":
        # 微调路径
        head_mult = None
        if bool(getattr(config.bert.finetuning, 'use_layered_lr', False)):
            head_mult = float(config.bert.finetuning.head_lr_multiplier)
        return build_optimizer_and_scheduler(
            model,
            total_steps=total_steps,
            base_lr=float(config.bert.finetuning.learning_rate),
            weight_decay=float(config.bert.finetuning.weight_decay),
            head_lr_multiplier=head_mult,
            warmup_steps=int(getattr(config.bert.finetuning, 'warmup_steps', 0) or 0),
            warmup_ratio=float(getattr(config.bert.finetuning, 'warmup_ratio', 0.1)),
        )
    else:
        raise ValueError("stage must be 'pretrain' or 'finetune'")


