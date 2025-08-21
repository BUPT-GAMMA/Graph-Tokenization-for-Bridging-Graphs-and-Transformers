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


def compute_multi_label_classification_metrics(
    y_true: np.ndarray, 
    y_score: np.ndarray
) -> Dict[str, float]:
    """
    计算多标签分类指标
    
    Args:
        y_true: [N, num_labels] 真实标签（0或1）
        y_score: [N, num_labels] 预测概率
        
    Returns:
        包含经典指标的字典：ap, accuracy, precision, recall, f1
    """
    assert y_true.shape == y_score.shape, f"标签和预测形状不匹配: {y_true.shape} vs {y_score.shape}"
    
    # 计算每个标签的AP
    label_aps = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            label_aps.append(ap)
        except ValueError:
            # 如果某个标签全为0或全为1，AP无法计算
            label_aps.append(0.0)
    
    # macro平均AP
    macro_ap = float(np.mean(label_aps))
    
    # 转换为预测标签
    y_pred = (y_score > 0.5).astype(int)
    
    # 计算经典指标
    # 使用macro平均（每个标签单独计算后平均）
    precisions, recalls, f1s = [], [], []
    for i in range(y_true.shape[1]):
        # 计算每个标签的precision, recall, f1
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    
    # 准确率使用Hamming accuracy（元素级准确率）
    accuracy = float(np.mean(y_true == y_pred))
    
    return {
        'ap': macro_ap,
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
    }


def compute_multi_target_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算多目标回归指标
    
    Args:
        y_true: [N, num_targets] 真实目标值
        y_pred: [N, num_targets] 预测目标值
        
    Returns:
        包含平均MAE等指标的字典
    """
    assert y_true.shape == y_pred.shape, f"标签和预测形状不匹配: {y_true.shape} vs {y_pred.shape}"
    
    # 计算每个目标的MAE
    target_maes = []
    target_mses = []
    for i in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        target_maes.append(mae)
        target_mses.append(mse)
    
    # 计算平均指标（使用标准名称）
    mae = float(np.mean(target_maes))
    mse = float(np.mean(target_mses))
    rmse = float(np.sqrt(mse))
    
    # 计算R2和相关系数（平均）
    target_r2s = []
    target_corrs = []
    for i in range(y_true.shape[1]):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        target_r2s.append(r2)
        target_corrs.append(corr if not np.isnan(corr) else 0.0)
    
    r2 = float(np.mean(target_r2s))
    correlation = float(np.mean(target_corrs))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
    }





