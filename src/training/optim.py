from __future__ import annotations

from typing import Optional
import torch


def build_optimizer_and_scheduler(
    model,
    *,
    total_steps: int,
    base_lr: float,
    weight_decay: float,
    head_lr_multiplier: Optional[float] = None,
    eta_min_ratio: float = 0.01,
):
    """
    构建 AdamW + CosineAnnealingLR；可选任务头学习率倍率。
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=base_lr * float(eta_min_ratio),
    )
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
            eta_min_ratio=1e-7 / max(float(config.bert.pretraining.learning_rate), 1e-12),
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
            eta_min_ratio=0.01,
        )
    else:
        raise ValueError("stage must be 'pretrain' or 'finetune'")


