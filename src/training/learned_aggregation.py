from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models.aggregators.variant_weighting import VariantWeightingAggregator


@dataclass
class AggregatorTrainConfig:
    hidden_dim: int = 256
    dropout: float = 0.1
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-2
    early_stopping_patience: int = 5
    use_pred_as_feat: bool = True  # 回归建议 True
    batch_size: int = 64  # 每批多少张图


class VariantSetDataset(Dataset):
    """将同一图的多变体样本组织为一个集合样本。

    回归：每条变体样本为 (pooled: [D], pred_norm: float)；标签为 y_true_norm: float
    分类：每条变体样本为 (pooled: [D], logits: [C])；标签为 y_true: int
    """

    def __init__(
        self,
        features_per_graph: List[np.ndarray],  # list of [K, D]
        aux_per_graph: List[np.ndarray],       # 回归: [K, 1] 预测(标准化); 分类: [K, C] logits
        labels: List[Any],                     # 回归: float(标准化); 分类: int
        task: Literal["regression", "classification"],
        use_pred_as_feat: bool = True,
    ):
        self.features = features_per_graph
        self.aux = aux_per_graph
        self.labels = labels
        self.task = task
        self.use_pred_as_feat = use_pred_as_feat

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feats = self.features[idx]   # [K, D]
        aux = self.aux[idx]
        label = self.labels[idx]
        return feats, aux, label


def _collate_variant_sets(batch: List[Tuple[np.ndarray, np.ndarray, Any]],
                          task: str,
                          use_pred_as_feat: bool) -> Dict[str, torch.Tensor]:
    """Pad 到同一 K，并生成 mask。"""
    Ks = [feats.shape[0] for feats, _, _ in batch]
    K_max = max(Ks)
    D = batch[0][0].shape[1]

    feats_padded = []
    mask = []
    if task == 'regression':
        # aux: [K, 1]
        aux_padded = []
    else:
        # aux: [K, C]
        C = batch[0][1].shape[1]
        aux_padded = []

    labels = []
    for feats, aux, label in batch:
        K = feats.shape[0]
        pad = K_max - K
        # pad feats
        if pad > 0:
            feats_pad = np.zeros((pad, D), dtype=np.float32)
            feats_cat = np.concatenate([feats.astype(np.float32), feats_pad], axis=0)
        else:
            feats_cat = feats.astype(np.float32)
        feats_padded.append(feats_cat)
        mask.append(np.array([True] * K + [False] * pad, dtype=bool))

        # pad aux
        if task == 'regression':
            if pad > 0:
                aux_pad = np.zeros((pad, 1), dtype=np.float32)
                aux_cat = np.concatenate([aux.astype(np.float32), aux_pad], axis=0)
            else:
                aux_cat = aux.astype(np.float32)
        else:
            if pad > 0:
                aux_pad = np.zeros((pad, C), dtype=np.float32)
                aux_cat = np.concatenate([aux.astype(np.float32), aux_pad], axis=0)
            else:
                aux_cat = aux.astype(np.float32)
        aux_padded.append(aux_cat)

        labels.append(label)

    feats_t = torch.from_numpy(np.stack(feats_padded, axis=0))  # [B, K, D]
    aux_t = torch.from_numpy(np.stack(aux_padded, axis=0))
    mask_t = torch.from_numpy(np.stack(mask, axis=0))  # [B, K]
    if task == 'regression':
        labels_t = torch.tensor(labels, dtype=torch.float32)
    else:
        labels_t = torch.tensor(labels, dtype=torch.long)

    return {
        'features': feats_t,
        'aux': aux_t,
        'mask': mask_t,
        'labels': labels_t,
    }


def _collect_variant_sets(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    task: Literal['regression', 'classification'],
    *,
    label_normalizer=None,
    use_pred_as_feat: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
    """冻结模型，收集每个图的变体特征与单独预测（或logits），并按图分组。"""
    model.eval()
    features_by_gid: Dict[int, List[np.ndarray]] = {}
    aux_by_gid: Dict[int, List[np.ndarray]] = {}
    label_by_gid: Dict[int, Any] = {}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            graph_ids = batch['graph_id'].cpu().numpy().tolist()

            outputs = model(input_ids, attention_mask, labels)
            pooled = outputs.get('pooled', None)
            if pooled is None:
                raise RuntimeError("模型输出缺少 'pooled'，请在Bert回归/分类模型的forward中加入该键")

            if task == 'regression':
                # 注意：这里的预测值是标准化空间的最终任务预测（TaskHead输出）
                # 修正：使用正确的字段名 'outputs'（而非 'predictions'）
                preds = outputs['outputs'].detach().to(torch.float32).cpu().numpy().reshape(-1, 1)
                pooled_np = pooled.detach().to(torch.float32).cpu().numpy()
                # 原始标签需要标准化以后用于监督
                if label_normalizer is None:
                    raise RuntimeError("回归聚合器训练需要 label_normalizer")
                orig = batch['original_label'].numpy().reshape(-1, 1)
                y_norm = np.array(label_normalizer.transform(orig.reshape(-1).tolist())).reshape(-1, 1)

                for gid, f, p, y in zip(graph_ids, pooled_np, preds, y_norm):
                    features_by_gid.setdefault(gid, []).append(f.astype(np.float32))
                    aux_by_gid.setdefault(gid, []).append(p.astype(np.float32))
                    label_by_gid[gid] = float(y[0])
            else:
                # 注意：这里的logits是标准化后的最终任务预测（TaskHead输出）
                # 修正：使用正确的字段名 'outputs'（而非 'logits'）
                logits = outputs['outputs'].detach().to(torch.float32).cpu().numpy()
                pooled_np = pooled.detach().to(torch.float32).cpu().numpy()
                y_true = labels.detach().cpu().numpy().reshape(-1)
                for gid, f, lg, y in zip(graph_ids, pooled_np, logits, y_true):
                    features_by_gid.setdefault(gid, []).append(f.astype(np.float32))
                    aux_by_gid.setdefault(gid, []).append(lg.astype(np.float32))
                    label_by_gid[gid] = int(y)

    # 整理为列表
    features_per_graph: List[np.ndarray] = []
    aux_per_graph: List[np.ndarray] = []
    labels: List[Any] = []
    for gid in features_by_gid.keys():
        feats = np.stack(features_by_gid[gid], axis=0)  # [K, D]
        aux = np.stack(aux_by_gid[gid], axis=0)         # [K, 1] or [K, C]
        features_per_graph.append(feats)
        aux_per_graph.append(aux)
        labels.append(label_by_gid[gid])

    return features_per_graph, aux_per_graph, labels


def train_variant_aggregator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    task: Literal['regression', 'classification'],
    *,
    label_normalizer=None,
    save_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> VariantWeightingAggregator:
    """训练变体加权聚合器。返回已训练的聚合器。"""
    if cfg is None:
        cfg = {}
    cfg_obj = AggregatorTrainConfig(**cfg)

    # 1) 收集 train / val 的图级集合数据
    train_feats, train_aux, train_labels = _collect_variant_sets(
        model, train_loader, device, task, label_normalizer=label_normalizer,
        use_pred_as_feat=cfg_obj.use_pred_as_feat,
    )
    val_feats, val_aux, val_labels = _collect_variant_sets(
        model, val_loader, device, task, label_normalizer=label_normalizer,
        use_pred_as_feat=cfg_obj.use_pred_as_feat,
    )

    # 2) 构建Dataset/DataLoader
    train_ds = VariantSetDataset(train_feats, train_aux, train_labels, task, cfg_obj.use_pred_as_feat)
    val_ds = VariantSetDataset(val_feats, val_aux, val_labels, task, cfg_obj.use_pred_as_feat)

    collate = lambda batch: _collate_variant_sets(batch, task, cfg_obj.use_pred_as_feat)
    train_dl = DataLoader(train_ds, batch_size=cfg_obj.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg_obj.batch_size, shuffle=False, collate_fn=collate)

    # 3) 创建聚合器
    D = train_feats[0].shape[1]
    if task == 'regression' and cfg_obj.use_pred_as_feat:
        input_dim = D + 1
    else:
        input_dim = D
    aggregator = VariantWeightingAggregator(input_dim=input_dim, hidden_dim=cfg_obj.hidden_dim,
                                            dropout=cfg_obj.dropout, use_pred_as_feat=cfg_obj.use_pred_as_feat)
    aggregator.to(device)

    optimizer = torch.optim.AdamW(aggregator.parameters(), lr=cfg_obj.lr, weight_decay=cfg_obj.weight_decay)

    if task == 'regression':
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 统一将验证指标视为“越小越好”：
    # 回归: MSE；分类: 1-accuracy
    best_val = float('inf')
    best_state = None
    patience = 0

    for epoch in range(cfg_obj.epochs):
        aggregator.train()
        train_loss_accum = 0.0
        train_steps = 0

        for batch in train_dl:
            feats = batch['features'].to(device)   # [B, K, D]
            aux = batch['aux'].to(device)         # [B, K, 1] or [B, K, C]
            mask = batch['mask'].to(device)       # [B, K]
            labels = batch['labels'].to(device)   # [B] or [B]

            if task == 'regression':
                # 拼接预测作为特征
                if cfg_obj.use_pred_as_feat:
                    feats = torch.cat([feats, aux], dim=-1)  # [B, K, D+1]
                weights = aggregator(feats, mask=mask)       # [B, K]
                # 标准化空间内聚合
                preds = aux.squeeze(-1)                      # [B, K]
                y_hat = (weights * preds).sum(dim=1)         # [B]
                loss = loss_fn(y_hat, labels.float())
            else:
                weights = aggregator(feats, mask=mask)       # [B, K]
                logits = aux                                  # [B, K, C]
                probs = torch.softmax(logits, dim=-1)        # [B, K, C]
                weights_exp = weights.unsqueeze(-1)          # [B, K, 1]
                p_agg = (weights_exp * probs).sum(dim=1)     # [B, C]
                loss = loss_fn(p_agg, labels.long())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            train_steps += 1

        # 验证
        aggregator.eval()
        val_metric_accum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_dl:
                feats = batch['features'].to(device)
                aux = batch['aux'].to(device)
                mask = batch['mask'].to(device)
                labels = batch['labels'].to(device)
                if task == 'regression':
                    if cfg_obj.use_pred_as_feat:
                        feats = torch.cat([feats, aux], dim=-1)
                    weights = aggregator(feats, mask=mask)
                    preds = aux.squeeze(-1)
                    y_hat = (weights * preds).sum(dim=1)
                    # 用 MSE 作验证指标（你也可以换 MAE）
                    metric = nn.functional.mse_loss(y_hat, labels.float())
                else:
                    weights = aggregator(feats, mask=mask)
                    logits = aux
                    probs = torch.softmax(logits, dim=-1)
                    p_agg = (weights.unsqueeze(-1) * probs).sum(dim=1)
                    # 验证用 1-accuracy 作为“损失型”指标便于早停（亦可直接accuracy并取最大）
                    pred_cls = torch.argmax(p_agg, dim=-1)
                    acc = (pred_cls == labels).float().mean()
                    metric = 1.0 - acc
                val_metric_accum += float(metric.detach().cpu().item())
                val_steps += 1

        val_metric = val_metric_accum / max(1, val_steps)
        # 早停：统一按最小化指标
        is_better = (val_metric < best_val)
        if is_better or best_state is None:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in aggregator.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg_obj.early_stopping_patience:
                break

    if best_state is not None:
        aggregator.load_state_dict(best_state)

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'state_dict': aggregator.state_dict(),
            'config': cfg_obj,
        }, os.path.join(save_dir, 'aggregator.pt'))

    return aggregator


