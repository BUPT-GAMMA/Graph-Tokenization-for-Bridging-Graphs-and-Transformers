from __future__ import annotations

from typing import Dict, Literal
from src.utils.logger import get_logger
import time
import numpy as np
import torch

from src.utils.metrics import compute_regression_metrics, compute_classification_metrics
logger = get_logger('tokenizerGraph.training.evaluate')


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    task: Literal["regression", "classification"],
    *,
    label_normalizer=None,
    aggregation_mode: Literal["avg", "best"] = "avg",
    epoch_num: int | None = None,
    total_epochs: int | None = None,
    log_style: Literal["online", "offline"] = "online",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    # 统一到 tokenizerGraph 命名空间的 logger
    start = time.time()
    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    next_percent_checkpoint = 10 if log_style == "offline" else None
    
    # 收集扁平化的所有预测、真实值和图ID
    all_preds, all_trues, all_gids = [], [], []

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        graph_ids = batch['graph_id'].cpu().numpy()

        outputs = model(input_ids, attention_mask, labels)
        # outputs 应该总是包含 loss 字段，这是模型设计保证的
        assert 'loss' in outputs, "模型输出缺少 loss 字段"
        loss = outputs['loss']
        total_loss += loss.item()
        steps += 1

        # offline 验证分段输出（online 由上层控制台/外部工具处理，这里不做 tqdm）
        if log_style == "offline" and steps_per_epoch:
            progress_pct = int(steps * 100 / max(1, steps_per_epoch))
            if next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                avg_loss = total_loss / steps
                elapsed = time.time() - start
                est_total = (elapsed / max(1, steps)) * steps_per_epoch
                eta = max(0.0, est_total - elapsed)
                epoch_prefix = f"Epoch {epoch_num}/{total_epochs} - " if epoch_num and total_epochs else ""
                logger.info(
                    f"[Offline] {epoch_prefix}Validation {progress_pct}% | loss={avg_loss:.4f} | elapsed={elapsed:.1f}s | eta={eta:.1f}s"
                )
                while next_percent_checkpoint is not None and progress_pct >= next_percent_checkpoint:
                    next_percent_checkpoint += 10

        if task == "regression":
            # 预测值是标准化的，在聚合前不要反向转换
            y_pred = outputs['predictions'].detach().cpu().numpy().reshape(-1)
            # original_label 未被标准化
            y_true = batch['original_label'].numpy().reshape(-1)
            all_preds.extend(y_pred.tolist())
            all_trues.extend(y_true.tolist())
            all_gids.extend(graph_ids.tolist())
        else: # classification
            logits = outputs['logits']
            y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
            y_true = labels.detach().cpu().numpy().reshape(-1)
            all_preds.extend(y_pred.tolist())
            all_trues.extend(y_true.tolist())
            all_gids.extend(graph_ids.tolist())

    avg_loss = total_loss / max(steps, 1)

    # 事后按graph_id分组
    grouped_preds = {}
    grouped_trues = {}
    for gid, pred, true in zip(all_gids, all_preds, all_trues):
        if gid not in grouped_preds:
            grouped_preds[gid] = []
            grouped_trues[gid] = true # 同一个gid的true label是相同的
        grouped_preds[gid].append(pred)

    # 聚合每个组的预测
    final_preds, final_trues = [], []
    for gid in grouped_preds:
        preds_for_gid = np.array(grouped_preds[gid])
        true_for_gid = grouped_trues[gid]
        
        if task == "regression":
            # 先反标准化，再聚合
            preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid.reshape(-1, 1))).flatten()

            if aggregation_mode == 'avg':
                agg_pred = np.mean(preds_for_gid_orig)
            elif aggregation_mode == 'best':
                errors = np.abs(preds_for_gid_orig - true_for_gid)
                agg_pred = preds_for_gid_orig[np.argmin(errors)]
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
        else: # classification
            if aggregation_mode == 'avg': # 投票
                agg_pred = np.bincount(preds_for_gid.astype(int)).argmax()
            elif aggregation_mode == 'best':
                correct_preds = preds_for_gid[preds_for_gid == true_for_gid]
                agg_pred = correct_preds[0] if len(correct_preds) > 0 else np.bincount(preds_for_gid.astype(int)).argmax()
            else:
                raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")

        final_preds.append(agg_pred)
        final_trues.append(true_for_gid)
    
    y_pred_agg = np.array(final_preds)
    y_true_agg = np.array(final_trues)
    
    metrics_out = {'val_loss': float(avg_loss)}
    if task == "regression":
        metrics = compute_regression_metrics(y_true_agg, y_pred_agg)
        metrics_out.update(metrics)
    else:
        metrics = compute_classification_metrics(y_true_agg, y_pred_agg)
        metrics_out.update(metrics)
        
    return metrics_out


