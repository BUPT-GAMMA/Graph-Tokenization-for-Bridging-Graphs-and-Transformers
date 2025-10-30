from __future__ import annotations

from typing import Dict, Literal, Optional, Any
from src.utils.logger import get_logger
import time
import numpy as np
import torch

from src.utils.metrics import (
    compute_regression_metrics, 
    compute_classification_metrics,
    compute_multi_label_classification_metrics,
    compute_multi_target_regression_metrics
)
logger = get_logger('tokenizerGraph.training.evaluate')


<<<<<<< HEAD
=======
def _learned_aggregation_core(
    aggregator, 
    pooled_list: list, 
    preds_or_probs: np.ndarray, 
    use_pred_as_feat: bool = False
) -> np.ndarray:
    """
    可学习聚合的核心逻辑
    
    Args:
        aggregator: 聚合器模型
        pooled_list: pooled特征列表 [K, feat_dim]
        preds_or_probs: 预测值或概率 [K, ...] 
        use_pred_as_feat: 是否将预测作为特征拼接
        
    Returns:
        加权聚合后的结果
    """
    # 边界检查：这些情况不应该发生，如果发生说明上层逻辑有问题
    if len(pooled_list) == 0:
        raise ValueError("pooled_list为空，上层逻辑错误")
    
    if len(pooled_list) != len(preds_or_probs):
        raise ValueError(f"pooled数量({len(pooled_list)})与预测数量({len(preds_or_probs)})不匹配，上层逻辑错误")
    
    pooled_arr = np.stack(pooled_list, axis=0).astype(np.float32)
    feats = pooled_arr
    
    if use_pred_as_feat:
        if preds_or_probs.ndim == 1:
            # 单目标回归：[K] -> [K, 1]
            pred_feats = preds_or_probs.reshape(-1, 1).astype(np.float32)
        else:
            # 多目标回归或分类：[K, D] -> [K, D]
            pred_feats = preds_or_probs.reshape(len(preds_or_probs), -1).astype(np.float32)
        feats = np.concatenate([pooled_arr, pred_feats], axis=1)
    
    feats_t = torch.from_numpy(feats).unsqueeze(0)  # [1, K, feat_dim]
    mask_t = torch.ones(1, feats_t.shape[1], dtype=torch.bool)  # [1, K]
    
    # 聚合器假设为有参数的神经网络，获取其设备
    agg_dev = next(aggregator.parameters()).device if hasattr(aggregator, 'parameters') else torch.device('cpu')
    feats_t, mask_t = feats_t.to(agg_dev), mask_t.to(agg_dev)
    
    with torch.no_grad():
        weights = aggregator(feats_t, mask=mask_t)  # [1, K]
        # 确保权重归一化（和为1）
        weights = torch.softmax(weights, dim=-1)
        
        # 转换预测到tensor并聚合
        # 确保preds_or_probs是numpy数组，如果是列表则转换
        if isinstance(preds_or_probs, list):
            preds_or_probs = np.array(preds_or_probs)
        preds_t = torch.from_numpy(preds_or_probs.astype(np.float32)).unsqueeze(0).to(agg_dev)
        
        if preds_or_probs.ndim == 1:
            # 单目标：[1, K] * [1, K] -> scalar
            result = (weights * preds_t).sum(dim=1).squeeze(0).item()
        else:
            # 多目标/多类别：[1, K] * [1, K, D] -> [1, D] -> [D]
            result = torch.einsum('bk,bkd->bd', weights, preds_t).squeeze(0).cpu().numpy()
        
    return result


>>>>>>> dev
def _aggregate_regression_predictions(preds_for_gid: np.ndarray, true_for_gid: np.ndarray, 
                                    aggregation_mode: str, label_normalizer, is_multi_target: bool) -> np.ndarray:
    """
    聚合回归预测结果
    
    Args:
        preds_for_gid: 该图ID的所有预测值
        true_for_gid: 该图ID的真实值
        aggregation_mode: 聚合模式 ('avg' 或 'best')
        label_normalizer: 标签归一化器
        is_multi_target: 是否为多目标回归
        
    Returns:
        聚合后的预测值
    """
    if is_multi_target:
        # 多目标回归：preds_for_gid形状应该是(K, num_targets)
        preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid))
        if aggregation_mode == 'avg':
            return np.mean(preds_for_gid_orig, axis=0)  # 对每个目标分别平均
        elif aggregation_mode == 'best':
            # 计算每个样本与真实值的距离（使用L2范数）
            errors = np.linalg.norm(preds_for_gid_orig - true_for_gid, axis=1)
            return preds_for_gid_orig[np.argmin(errors)]
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
    else:
        # 单目标回归：preds_for_gid形状应该是(K,)
        preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid.reshape(-1, 1))).flatten()
        if aggregation_mode == 'avg':
            return np.mean(preds_for_gid_orig)
        elif aggregation_mode == 'best':
            errors = np.abs(preds_for_gid_orig - true_for_gid)
            return preds_for_gid_orig[np.argmin(errors)]
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")


def _aggregate_classification_predictions(preds_for_gid: np.ndarray, logits_for_gid: list, 
                                        true_for_gid: np.ndarray, aggregation_mode: str, 
                                        is_multi_label: bool) -> tuple:
    """
    聚合分类预测结果
    
    Args:
        preds_for_gid: 该图ID的所有预测类别
        logits_for_gid: 该图ID的所有概率/logits
        true_for_gid: 该图ID的真实值
        aggregation_mode: 聚合模式
        is_multi_label: 是否为多标签分类
        
    Returns:
        (聚合后的预测值, 聚合后的概率)
    """
    if is_multi_label:
        # 多标签分类：preds_for_gid是概率数组，直接聚合
        if aggregation_mode in ['avg', 'learned']:
            agg_pred = np.mean(preds_for_gid, axis=0)  # 概率平均
        elif aggregation_mode == 'best':
            # 选择与真实标签最接近的预测
            errors = np.mean(np.abs(preds_for_gid - true_for_gid), axis=1)
            agg_pred = preds_for_gid[np.argmin(errors)]
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
        return agg_pred, None
    else:
        # 单标签分类：preds_for_gid是类别索引，有单独的概率数据
        logits_mat = np.array(logits_for_gid, dtype=np.float32)
        
        if len(logits_mat) == 0:
<<<<<<< HEAD
            raise ValueError(f"缺少概率数据，无法计算AUC/AP指标")
        
        if aggregation_mode == 'avg':
=======
            raise ValueError("缺少概率数据，无法计算AUC/AP指标")
        
        if aggregation_mode in ['avg', 'learned']:
>>>>>>> dev
            # 概率平均后argmax
            p_agg = np.mean(logits_mat, axis=0)
            agg_pred = int(np.argmax(p_agg))
            agg_prob_np = p_agg
        elif aggregation_mode == 'best':
            # 选择正确预测对应的概率，如果没有则使用平均概率
            correct_indices = np.where(preds_for_gid == true_for_gid)[0]
            if len(correct_indices) > 0:
                agg_pred = int(preds_for_gid[correct_indices[0]])
                agg_prob_np = logits_mat[correct_indices[0]]
            else:
                agg_pred = np.bincount(preds_for_gid.astype(int)).argmax()
                agg_prob_np = np.mean(logits_mat, axis=0)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation_mode}")
        
        return agg_pred, agg_prob_np


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
    task: Literal["mlm", "regression", "classification", "multi_label_classification", "multi_target_regression"],
    *,
    task_handler,
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

        outputs = model(input_ids, attention_mask)
        loss = task_handler.compute_loss(outputs['outputs'], labels)
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

        # 使用task_handler统一获取预测和概率
        if task_handler.is_regression_task():
            # 预测值是标准化的，在聚合前不要反向转换
            y_pred = task_handler.get_predictions(outputs['outputs'])  # 已经是numpy数组
            # original_label 未被标准化
            y_true = batch['original_label'].numpy()
            if y_pred.ndim > 1:  # 多目标回归
                all_preds.extend(y_pred.tolist())
                all_trues.extend(y_true.tolist())
            else:  # 单目标回归
                y_pred = y_pred.reshape(-1)
                y_true = y_true.reshape(-1)
                all_preds.extend(y_pred.tolist())
                all_trues.extend(y_true.tolist())
            all_gids.extend(graph_ids.tolist())
            # 采集序列池化向量
            if 'pooled' in outputs:
                # bfloat16 -> float32 再转 numpy
                pooled_np = outputs['pooled'].detach().to(torch.float32).cpu().numpy()
                # 与 y_pred 一一对应
                for row in pooled_np:
                    all_pooled.append(row)
        else: # classification tasks  
            y_true = labels.detach().cpu().numpy()
            all_trues.extend(y_true.tolist())
            all_gids.extend(graph_ids.tolist())
            
            if task_handler.is_multi_label():
                # 多标签：直接用概率计算AP
                probs = task_handler.get_probabilities(outputs['outputs'])
                all_logits.extend(probs.tolist())
            else:
                # 单标签：用类别索引计算accuracy，用概率计算AUC
                preds = task_handler.get_predictions(outputs['outputs'])
                probs = task_handler.get_probabilities(outputs['outputs'])
                all_preds.extend(preds.reshape(-1).tolist())
                all_logits.extend(probs.tolist())
                
            if 'pooled' in outputs:
                pooled_np = outputs['pooled'].detach().to(torch.float32).cpu().numpy()
                for row in pooled_np:
                    all_pooled.append(row)

    avg_loss = total_loss / max(steps, 1)

    # 数据完整性检查
    expected_length = len(all_gids)
    if len(all_pooled) != expected_length:
        logger.warning(f"pooled数据不完整: 期望{expected_length}, 实际{len(all_pooled)}")

    # 事后按graph_id分组
    grouped_preds = {}
    grouped_trues = {}
    grouped_pooled = {}
    grouped_logits = {}  # 仅分类任务
<<<<<<< HEAD
    # 为了与 all_pooled 对齐，需要同时迭代 pooled 索引
    if task_handler.is_regression_task():
        pooled_iter = iter(all_pooled)
        for gid, pred, true in zip(all_gids, all_preds, all_trues):
=======
    # 统一使用索引方式处理pooled数据对齐
    if task_handler.is_regression_task():
        for i, (gid, pred, true) in enumerate(zip(all_gids, all_preds, all_trues)):
>>>>>>> dev
            if gid not in grouped_preds:
                grouped_preds[gid] = []
                grouped_trues[gid] = true # 同一个gid的true label是相同的
                grouped_pooled[gid] = []
            grouped_preds[gid].append(pred)
            # 如果存在 pooled，按索引取出
            if i < len(all_pooled):
                grouped_pooled[gid].append(all_pooled[i])
            else:
                logger.warning(f"回归任务 gid={gid} 缺少pooled数据，索引={i}, all_pooled长度={len(all_pooled)}")
    else:
        # 分类任务聚合
<<<<<<< HEAD
        pooled_iter = iter(all_pooled)
        
=======
>>>>>>> dev
        if task_handler.is_multi_label():
            # 多标签：只有概率，没有预测类别
            # 确保数据长度一致
            assert len(all_gids) == len(all_trues) == len(all_logits), \
                f"多标签数据长度不一致: gids={len(all_gids)}, trues={len(all_trues)}, logits={len(all_logits)}"
            
            for i, (gid, true, logit) in enumerate(zip(all_gids, all_trues, all_logits)):
                if gid not in grouped_trues:
                    grouped_trues[gid] = true
                    grouped_pooled[gid] = []
                    grouped_logits[gid] = []
                grouped_logits[gid].append(logit)
                # 处理pooled数据（如果存在）
                if i < len(all_pooled):
                    grouped_pooled[gid].append(all_pooled[i])
<<<<<<< HEAD
=======
                else:
                    logger.warning(f"多标签分类 gid={gid} 缺少pooled数据，索引={i}, all_pooled长度={len(all_pooled)}")
>>>>>>> dev
        else:
            # 单标签：有预测类别和概率
            # 确保所有数据长度一致
            assert len(all_gids) == len(all_preds) == len(all_trues) == len(all_logits), \
                f"数据长度不一致: gids={len(all_gids)}, preds={len(all_preds)}, trues={len(all_trues)}, logits={len(all_logits)}"
            
            for i, (gid, pred, true, logit) in enumerate(zip(all_gids, all_preds, all_trues, all_logits)):
                if gid not in grouped_preds:
                    grouped_preds[gid] = []
                    grouped_trues[gid] = true
                    grouped_pooled[gid] = []
                    grouped_logits[gid] = []
                grouped_preds[gid].append(pred)
                grouped_logits[gid].append(logit)
                # 处理pooled数据（如果存在）
                if i < len(all_pooled):
                    grouped_pooled[gid].append(all_pooled[i])
<<<<<<< HEAD
=======
                else:
                    logger.warning(f"单标签分类 gid={gid} 缺少pooled数据，索引={i}, all_pooled长度={len(all_pooled)}")
>>>>>>> dev

    # 聚合每个组的预测
    final_preds, final_trues = [], []
    final_scores = []  # 若可得，保存按 gid 聚合后的概率分布（用于 AUC/AP）
    
    # 确定聚合的键集合（回归和单标签分类用grouped_preds，多标签用grouped_trues）
    gid_keys = grouped_preds if not task_handler.is_multi_label() else grouped_trues
    
    for gid in gid_keys:
        true_for_gid = grouped_trues[gid]
        
        # 根据任务类型获取预测数据
        if task_handler.is_multi_label():
            # 多标签：只有概率数据
            preds_for_gid = np.array(grouped_logits[gid])
        else:
            # 回归和单标签分类：有预测数据
            preds_for_gid = np.array(grouped_preds[gid])
        
        if task_handler.is_regression_task():
            if aggregation_mode == 'learned' and aggregator is not None:
                # 使用可学习聚合（在标准化空间加权，再反标准化）
                pooled_list = grouped_pooled.get(gid, [])
                
                if task_handler.is_multi_target():
                    # 多目标回归的learned aggregation
                    if len(pooled_list) != len(preds_for_gid):
<<<<<<< HEAD
                        # 回退到avg
=======
                        # 回退到avg，并给出警告
                        logger.warning(
                            f"[learned->avg fallback] 多目标回归 gid={gid}: pooled数={len(pooled_list)} 与 预测数={len(preds_for_gid)} 不一致"
                        )
>>>>>>> dev
                        preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid))
                        agg_pred = np.mean(preds_for_gid_orig, axis=0)  # 对每个目标分别平均
                    else:
                        # 多目标learned aggregation: 对每个目标分别计算权重聚合
<<<<<<< HEAD
                        pooled_arr = np.stack(pooled_list, axis=0).astype(np.float32)
                        feats = pooled_arr
                        # 若聚合器声明使用预测作为特征，则拼接（展平预测向量）
                        if getattr(aggregator, 'use_pred_as_feat', False):
                            preds_flat = preds_for_gid.reshape(len(preds_for_gid), -1).astype(np.float32)
                            feats = np.concatenate([pooled_arr, preds_flat], axis=1)
                        feats_t = torch.from_numpy(feats).unsqueeze(0)
                        mask_t = torch.ones(1, feats_t.shape[1], dtype=torch.bool)
                        agg_dev = next(aggregator.parameters()).device if hasattr(aggregator, 'parameters') else torch.device('cpu')
                        feats_t = feats_t.to(agg_dev)
                        mask_t = mask_t.to(agg_dev)
                        with torch.no_grad():
                            weights = aggregator(feats_t, mask=mask_t)  # [1, K]
                            # 对多目标预测进行加权聚合
                            preds_t = torch.from_numpy(preds_for_gid.astype(np.float32)).unsqueeze(0).to(agg_dev)  # [1, K, num_targets]
                            y_hat_norm = torch.einsum('bk,bkd->bd', weights, preds_t).squeeze(0).cpu().numpy()  # [num_targets]
=======
                        use_pred_feat = getattr(aggregator, 'use_pred_as_feat', False)
                        y_hat_norm = _learned_aggregation_core(aggregator, pooled_list, preds_for_gid, use_pred_feat)
>>>>>>> dev
                        # 反标准化
                        inv = label_normalizer.inverse_transform([y_hat_norm])
                        agg_pred = np.array(inv[0])
                else:
                    # 单目标回归的learned aggregation（原逻辑）
                    preds_norm = preds_for_gid.reshape(-1)
                    if len(pooled_list) != len(preds_norm):
<<<<<<< HEAD
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
=======
                        # 回退到avg，并给出警告
                        logger.warning(
                            f"[learned->avg fallback] 单目标回归 gid={gid}: pooled数={len(pooled_list)} 与 预测数={len(preds_norm)} 不一致"
                        )
                        preds_for_gid_orig = np.array(label_normalizer.inverse_transform(preds_for_gid.reshape(-1, 1))).flatten()
                        agg_pred = np.mean(preds_for_gid_orig)
                    else:
                        use_pred_feat = getattr(aggregator, 'use_pred_as_feat', False)
                        y_hat_norm = _learned_aggregation_core(aggregator, pooled_list, preds_norm, use_pred_feat)
>>>>>>> dev
                        # 反标准化：本项目的 LabelNormalizer.inverse_transform 返回 List[float]
                        inv = label_normalizer.inverse_transform([float(y_hat_norm)])
                        assert isinstance(inv, list) and len(inv) == 1, "LabelNormalizer.inverse_transform 应返回长度为1的list"
                        agg_pred = float(inv[0])
<<<<<<< HEAD
=======
            elif aggregation_mode == 'learned' and aggregator is None:
                # 积极检查：learned模式但缺少聚合器
                raise ValueError(f"aggregation_mode='learned' 但未提供 aggregator，gid={gid}")
>>>>>>> dev
            else:
                # 使用辅助函数进行聚合
                agg_pred = _aggregate_regression_predictions(
                    preds_for_gid, true_for_gid, aggregation_mode, 
                    label_normalizer, task_handler.is_multi_target()
                )
        else: # classification
<<<<<<< HEAD
            # 使用辅助函数进行分类聚合
            logits_for_gid = grouped_logits.get(gid, [])
            agg_pred, agg_prob_np = _aggregate_classification_predictions(
                preds_for_gid, logits_for_gid, true_for_gid, 
                aggregation_mode, task_handler.is_multi_label()
            )
=======
            if aggregation_mode == 'learned' and aggregator is not None:
                # 分类任务的learned聚合
                pooled_list = grouped_pooled.get(gid, [])
                
                if task_handler.is_multi_label():
                    # 多标签分类learned聚合
                    probs_for_gid = np.array(grouped_logits[gid])  # [K, num_labels]
                    
                    if len(pooled_list) != len(probs_for_gid):
                        # 回退到avg并警告
                        logger.warning(
                            f"[learned->avg fallback] 多标签分类 gid={gid}: pooled数={len(pooled_list)} 与 概率数={len(probs_for_gid)} 不一致"
                        )
                        agg_pred = np.mean(probs_for_gid, axis=0)
                        agg_prob_np = None
                    else:
                        # learned聚合
                        use_pred_feat = getattr(aggregator, 'use_pred_as_feat', False)
                        agg_pred = _learned_aggregation_core(aggregator, pooled_list, probs_for_gid, use_pred_feat)
                        agg_prob_np = None
                else:
                    # 单标签分类learned聚合
                    probs_for_gid = np.array(grouped_logits[gid])  # [K, num_classes]
                    
                    if len(pooled_list) != len(probs_for_gid):
                        # 回退到avg并警告
                        logger.warning(
                            f"[learned->avg fallback] 单标签分类 gid={gid}: pooled数={len(pooled_list)} 与 概率数={len(probs_for_gid)} 不一致"
                        )
                        p_agg = np.mean(probs_for_gid, axis=0)
                        agg_pred = int(np.argmax(p_agg))
                        agg_prob_np = p_agg
                    else:
                        # learned聚合
                        use_pred_feat = getattr(aggregator, 'use_pred_as_feat', False)
                        p_agg = _learned_aggregation_core(aggregator, pooled_list, probs_for_gid, use_pred_feat)
                        agg_pred = int(np.argmax(p_agg))
                        agg_prob_np = p_agg
            elif aggregation_mode == 'learned' and aggregator is None:
                # 积极检查：learned模式但缺少聚合器
                raise ValueError(f"aggregation_mode='learned' 但未提供 aggregator，gid={gid}")
            else:
                # 使用辅助函数进行分类聚合（原有逻辑）
                logits_for_gid = grouped_logits.get(gid, [])
                agg_pred, agg_prob_np = _aggregate_classification_predictions(
                    preds_for_gid, logits_for_gid, true_for_gid, 
                    aggregation_mode, task_handler.is_multi_label()
                )
>>>>>>> dev

        final_preds.append(agg_pred)
        final_trues.append(true_for_gid)
        
        # 保存概率数据用于指标计算
        if task_handler.is_multi_label():
            # 多标签：agg_pred本身就是概率
            final_scores.append(agg_pred.tolist())
        elif task_handler.is_classification_task():
            # 单标签：使用单独的概率数据
            if agg_prob_np is None:
                raise ValueError(f"Graph ID {gid} 缺少概率数据")
            final_scores.append(agg_prob_np.tolist())
    
    y_pred_agg = np.array(final_preds)
    y_true_agg = np.array(final_trues)
    
    metrics_out = {'val_loss': float(avg_loss)}
    
    if task_handler.is_multi_target():
        # 多目标回归：y_true_agg和y_pred_agg都是[N, num_targets]格式
        metrics = compute_multi_target_regression_metrics(y_true_agg, y_pred_agg)
        metrics_out.update(metrics)
    elif task_handler.is_regression_task():
        # 单目标回归
        metrics = compute_regression_metrics(y_true_agg, y_pred_agg)
        metrics_out.update(metrics)
    elif task_handler.is_multi_label():
        # 多标签分类：y_true_agg和y_pred_agg都是[N, num_labels]格式
        y_score_agg = np.array(final_scores) if len(final_scores) == len(y_true_agg) and len(final_scores) > 0 else None
        if y_score_agg is not None:
            metrics = compute_multi_label_classification_metrics(y_true_agg, y_score_agg)
            metrics_out.update(metrics)
        else:
            logger.warning("多标签分类缺少概率分数，无法计算AP指标")
    elif task_handler.is_classification_task():
        # 单标签分类
        y_score_agg = np.array(final_scores) if len(final_scores) == len(y_true_agg) and len(final_scores) > 0 else None
        metrics = compute_classification_metrics(y_true_agg, y_pred_agg, y_score=y_score_agg)
        metrics_out.update(metrics)
    else:
        raise ValueError(f"不支持的任务类型: {task}")
        
    return metrics_out


