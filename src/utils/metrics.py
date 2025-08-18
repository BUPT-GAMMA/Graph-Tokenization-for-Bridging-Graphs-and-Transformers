from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'correlation': float(corr),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    # 默认不计算，若提供了概率分布再计算 AUC/AP，避免在多分类下误用离散标签
    roc_auc: Optional[float] = None
    ap: Optional[float] = None

    try:
        if y_score is not None:
            y_true = y_true.astype(int)
            classes = np.unique(y_true)
            if y_score.ndim == 1:  # 形如 [N]，视作正类概率
                # 二分类：需要正类概率
                roc_auc = float(roc_auc_score(y_true, y_score))
                ap = float(average_precision_score(y_true, y_score))
            else:
                num_classes = y_score.shape[1]
                if num_classes == 2:
                    # 二分类：使用正类列
                    roc_auc = float(roc_auc_score(y_true, y_score[:, 1]))
                    ap = float(average_precision_score(y_true, y_score[:, 1]))
                elif len(classes) > 2 and num_classes > 2:
                    # 多分类：需要 one-vs-rest 概率矩阵
                    roc_auc = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'))
                    # AP 多分类：对 one-hot 进行 macro 平均
                    y_true_oh = label_binarize(y_true, classes=np.arange(num_classes))
                    ap = float(average_precision_score(y_true_oh, y_score, average='macro'))
    except Exception:
        # 任何异常下，保持 None，不中断主流程
        roc_auc = None
        ap = None

    out: Dict[str, float] = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
    }
    if roc_auc is not None:
        out['roc_auc'] = float(roc_auc)
    if ap is not None:
        out['ap'] = float(ap)
    return out





