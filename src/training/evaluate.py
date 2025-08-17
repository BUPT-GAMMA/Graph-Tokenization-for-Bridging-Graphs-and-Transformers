from __future__ import annotations

from typing import Dict, Literal, Optional, Any
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
    aggregation_mode: Literal["avg", "best", "learned"] = "avg",
    epoch_num: int | None = None,
    total_epochs: int | None = None,
    log_style: Literal["online", "offline"] = "online",
    aggregator: Optional[Any] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    steps = 0
    # 统一到 tokenizerGraph 命名空间的 logger
    start = time.time()
    steps_per_epoch = len(dataloader) if hasattr(dataloader, "__len__") else None
    next_percent_checkpoint = 10 if log_style == "offline" else None
    
    # 收集扁平化的所有预测、真实值、图ID及可选特征
    all_preds, all_trues, all_gids = [], [], []
    all_pooled = []
    all_logits = []  # 仅分类任务使用

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
            # 采集序列池化向量
            if 'pooled' in outputs:
                pooled_np = outputs['pooled'].detach().cpu().numpy()
                # 与 y_pred 一一对应
                for row in pooled_np:
                    all_pooled.append(row)
        else: # classification
            logits = outputs['logits']
            y_pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
            y_true = labels.detach().cpu().numpy().reshape(-1)
            all_preds.extend(y_pred.tolist())
            all_trues.extend(y_true.tolist())
            all_gids.extend(graph_ids.tolist())
            # 采集 logits 与 pooled
            all_logits.extend(logits.detach().cpu().numpy().tolist())
            if 'pooled' in outputs:
                pooled_np = outputs['pooled'].detach().cpu().numpy()
                for row in pooled_np:
                    all_pooled.append(row)

    avg_loss = total_loss / max(steps, 1)

    # 事后按graph_id分组
    grouped_preds = {}
    grouped_trues = {}
    grouped_pooled = {}
    grouped_logits = {}  # 仅分类任务
    # 为了与 all_pooled 对齐，需要同时迭代 pooled 索引
    if task == "regression":
        pooled_iter = iter(all_pooled)
        for gid, pred, true in zip(all_gids, all_preds, all_trues):
            if gid not in grouped_preds:
                grouped_preds[gid] = []
                grouped_trues[gid] = true # 同一个gid的true label是相同的
                grouped_pooled[gid] = []
            grouped_preds[gid].append(pred)
            # 如果存在 pooled，依次取出
            try:
                grouped_pooled[gid].append(next(pooled_iter))
            except StopIteration:
                pass
    else:
        # 分类：需要与 logits 对齐
        pooled_iter = iter(all_pooled)
        logits_iter = iter(all_logits)
        for gid, pred, true in zip(all_gids, all_preds, all_trues):
            if gid not in grouped_preds:
                grouped_preds[gid] = []
                grouped_trues[gid] = true
                grouped_pooled[gid] = []
                grouped_logits[gid] = []
            grouped_preds[gid].append(pred)
            try:
                grouped_pooled[gid].append(next(pooled_iter))
            except StopIteration:
                pass
            try:
                grouped_logits[gid].append(next(logits_iter))
            except StopIteration:
                pass
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
            if aggregation_mode == 'learned' and aggregator is not None:
                # 使用可学习聚合（在标准化空间加权，再反标准化）
                preds_norm = preds_for_gid.reshape(-1)
                pooled_list = grouped_pooled.get(gid, [])
                if len(pooled_list) != len(preds_norm):
                    # 回退到avg
                    preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid.reshape(-1, 1))).flatten()
                    agg_pred = np.mean(preds_for_gid_orig)
                else:
                    pooled_arr = np.stack(pooled_list, axis=0).astype(np.float32)
                    feats = pooled_arr
                    # 若聚合器声明使用预测作为特征，则拼接
                    if getattr(aggregator, 'use_pred_as_feat', False):
                        feats = np.concatenate([pooled_arr, preds_norm.reshape(-1, 1).astype(np.float32)], axis=1)
                    feats_t = torch.from_numpy(feats).unsqueeze(0)
                    mask_t = torch.ones(1, feats_t.shape[1], dtype=torch.bool)
                    agg_dev = next(aggregator.parameters()).device if hasattr(aggregator, 'parameters') else torch.device('cpu')
                    feats_t = feats_t.to(agg_dev)
                    mask_t = mask_t.to(agg_dev)
                    with torch.no_grad():
                        weights = aggregator(feats_t, mask=mask_t)  # [1, K]
                        preds_t = torch.from_numpy(preds_norm.astype(np.float32)).unsqueeze(0).to(agg_dev)
                        y_hat_norm = (weights * preds_t).sum(dim=1).squeeze(0).item()
                    # 反标准化：本项目的 LabelNormalizer.inverse_transform 返回 List[float]
                    inv = label_normalizer.inverse_transform([float(y_hat_norm)])
                    assert isinstance(inv, list) and len(inv) == 1, "LabelNormalizer.inverse_transform 应返回长度为1的list"
                    agg_pred = float(inv[0])
            else:
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
            if aggregation_mode == 'learned' and aggregator is not None:
                logits_mat = np.array(grouped_logits.get(gid, []), dtype=np.float32)
                pooled_list = grouped_pooled.get(gid, [])
                if logits_mat.shape[0] == 0 or len(pooled_list) != logits_mat.shape[0]:
                    # 回退到投票
                    agg_pred = np.bincount(preds_for_gid.astype(int)).argmax()
                else:
                    feats = np.stack(pooled_list, axis=0).astype(np.float32)
                    feats_t = torch.from_numpy(feats).unsqueeze(0)
                    mask_t = torch.ones(1, feats_t.shape[1], dtype=torch.bool)
                    agg_dev = next(aggregator.parameters()).device if hasattr(aggregator, 'parameters') else torch.device('cpu')
                    feats_t = feats_t.to(agg_dev)
                    mask_t = mask_t.to(agg_dev)
                    with torch.no_grad():
                        weights = aggregator(feats_t, mask=mask_t)  # [1, K]
                        logits_t = torch.from_numpy(logits_mat).unsqueeze(0).to(agg_dev)  # [1, K, C]
                        probs_t = torch.softmax(logits_t, dim=-1)  # [1, K, C]
                        weights_exp = weights.unsqueeze(-1)  # [1, K, 1]
                        p_agg = (weights_exp * probs_t).sum(dim=1).squeeze(0)  # [C]
                        agg_pred = int(torch.argmax(p_agg).item())
            else:
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


