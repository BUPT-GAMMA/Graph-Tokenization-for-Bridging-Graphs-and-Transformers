#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证可从项目根执行
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.models.bert.heads import create_model_from_udi


def main():
    assert torch.cuda.is_available(), "必须使用GPU执行模型相关计算"
    device = torch.device("cuda")

    cfg = ProjectConfig()
    # 使用 ZINC 数据集 + BERT 编码器 (重构后统一架构)
    cfg.dataset.name = "zinc"
    cfg.encoder.type = "bert"  # 🆕 使用新的配置方式
    cfg.task.type = "regression"  # 🆕 明确指定任务类型
    
    # 加速 smoke：小批量+少步
    cfg.bert.finetuning.batch_size = 4
    cfg.bert.finetuning.epochs = 1
    cfg.serialization.multiple_sampling.enabled = False

    # UDI: 加载带 graph_id 的数据
    udi = UnifiedDataInterface(cfg, cfg.dataset.name)

    # 🆕 使用重构后的统一模型创建接口
    model, task_handler = create_model_from_udi(
        udi=udi, 
        pretrained_model=None,  # 使用随机初始化，避免预训练加载问题
        pooling_method='mean'
    )
    model = model.to(device)
    model.train()
    
    print(f"🎯 模型信息:")
    print(f"  - 类型: {type(model).__name__}")
    print(f"  - 编码器: {type(model.encoder).__name__}")
    print(f"  - 任务类型: {model.task_type}")
    print(f"  - 输出维度: {model.output_dim}")
    print(f"  - 损失函数: {type(task_handler.loss_fn).__name__}")

    # 🆕 取一小批数据做前向+反传 - 使用重构后的简化接口
    from src.training.tasks import build_regression_loaders
    
    train_dl, val_dl, test_dl, normalizer = build_regression_loaders(cfg, udi, cfg.serialization.method)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    print("🚀 开始ZINC统一架构smoke训练(1个epoch)...")
    for step, batch in enumerate(train_dl):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 🆕 使用统一模型接口
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 🆕 使用TaskHandler计算损失
        loss = task_handler.compute_loss(outputs['outputs'], labels)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if (step + 1) % 10 == 0:
            print(f"step {step+1}: loss={float(loss):.4f}, batch={input_ids.shape}")
        if step >= 20:
            break

    print("✅ smoke 完成：前向+反传均成功，模型能正常输出。")


if __name__ == "__main__":
    main()


