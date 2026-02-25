from __future__ import annotations

from typing import Dict, Optional, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.tensorboard import SummaryWriter


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

    # Only compute AUC/AP when probability scores are provided
    roc_auc: Optional[float] = None
    ap: Optional[float] = None

    try:
        if y_score is not None:
            y_true = y_true.astype(int)
            classes = np.unique(y_true)
            if y_score.ndim == 1:  # shape [N], treat as positive class probability
                # Binary classification
                roc_auc = float(roc_auc_score(y_true, y_score))
                ap = float(average_precision_score(y_true, y_score))
            else:
                num_classes = y_score.shape[1]
                if num_classes == 2:
                    # Binary: use positive class column
                    roc_auc = float(roc_auc_score(y_true, y_score[:, 1]))
                    ap = float(average_precision_score(y_true, y_score[:, 1]))
                elif len(classes) > 2 and num_classes > 2:
                    # Multi-class: one-vs-rest probability matrix
                    roc_auc = float(roc_auc_score(y_true, y_score, multi_class='ovr', average='macro'))
                    # AP multi-class: macro average over one-hot
                    y_true_oh = label_binarize(y_true, classes=np.arange(num_classes))
                    ap = float(average_precision_score(y_true_oh, y_score, average='macro'))
    except Exception:
        # On any exception, keep None, don't interrupt main flow
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
    Compute multi-label classification metrics.
    
    Args:
        y_true: [N, num_labels] ground truth (0 or 1)
        y_score: [N, num_labels] predicted probabilities
        
    Returns:
        Dict with ap, accuracy, precision, recall, f1
    """
    assert y_true.shape == y_score.shape, f"Label and prediction shape mismatch: {y_true.shape} vs {y_score.shape}"
    
    # Per-label AP
    label_aps = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            label_aps.append(ap)
        except ValueError:
            # AP undefined when a label is all-0 or all-1
            label_aps.append(0.0)
    
    # Macro-average AP
    macro_ap = float(np.mean(label_aps))
    
    # Convert to predicted labels
    y_pred = (y_score > 0.5).astype(int)
    
    # Compute standard metrics (macro average across labels)
    precisions, recalls, f1s = [], [], []
    for i in range(y_true.shape[1]):
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    
    # Hamming accuracy (element-wise)
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
    Compute multi-target regression metrics.
    
    Args:
        y_true: [N, num_targets] ground truth
        y_pred: [N, num_targets] predictions
        
    Returns:
        Dict with averaged mae, mse, rmse, r2, correlation
    """
    assert y_true.shape == y_pred.shape, f"Label and prediction shape mismatch: {y_true.shape} vs {y_pred.shape}"
    
    # Per-target MAE
    target_maes = []
    target_mses = []
    for i in range(y_true.shape[1]):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        target_maes.append(mae)
        target_mses.append(mse)
    
    # Average metrics
    mae = float(np.mean(target_maes))
    mse = float(np.mean(target_mses))
    rmse = float(np.sqrt(mse))
    
    # Average R2 and correlation
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



def add_metrics_to_writer(writer: SummaryWriter, base: str, metrics: Dict[str, float], task: str):
    if task == "regression":
      assert 'mae' in metrics, "Regression task requires MAE"
      writer.add_scalar(f'{base}/MAE', float(metrics['mae']))
      if 'mse' in metrics:
        writer.add_scalar(f'{base}/MSE', float(metrics['mse']))
      if 'r2' in metrics:
        writer.add_scalar(f'{base}/R2', float(metrics['r2']))
    elif task == "classification":
      if 'accuracy' in metrics:
        writer.add_scalar(f'{base}/Accuracy', float(metrics['accuracy']))
      if 'f1' in metrics:
        writer.add_scalar(f'{base}/F1', float(metrics['f1']))
      if 'roc_auc' in metrics:
        writer.add_scalar(f'{base}/ROC_AUC', float(metrics['roc_auc']))
      if 'ap' in metrics:
        writer.add_scalar(f'{base}/AP', float(metrics['ap']))
    elif task == "multi_label_classification":
      assert 'ap' in metrics, "Multi-label classification requires AP"
      writer.add_scalar(f'{base}/AP', float(metrics['ap']))
    elif task == "multi_target_regression":
      assert 'mae' in metrics, "Multi-target regression requires MAE"
      writer.add_scalar(f'{base}/MAE', float(metrics['mae']))
      if 'mse' in metrics:
        writer.add_scalar(f'{base}/MSE', float(metrics['mse']))
      if 'r2' in metrics:
        writer.add_scalar(f'{base}/R2', float(metrics['r2']))
    else:
        raise ValueError(f"Unsupported task type: {task}")
def log_wandb_metrics(wandb_logger: Any,base: str, metrics: Dict[str, float], task: str):
    if wandb_logger is None:
       return
    
    wb_payload = {}
    
    if task == "regression":
      assert 'mae' in metrics, "Regression task requires MAE"
      wb_payload[f'{base}/MAE'] = float(metrics['mae'])
      if 'mse' in metrics:
        wb_payload[f'{base}/MSE'] = float(metrics['mse'])
      if 'r2' in metrics:
        wb_payload[f'{base}/R2'] = float(metrics['r2'])
    elif task == "classification":
      if 'accuracy' in metrics:
        wb_payload[f'{base}/Accuracy'] = float(metrics['accuracy'])
      if 'f1' in metrics:
        wb_payload[f'{base}/F1'] = float(metrics['f1'])
      if 'roc_auc' in metrics:
        wb_payload[f'{base}/ROC_AUC'] = float(metrics['roc_auc'])
      if 'ap' in metrics:
        wb_payload[f'{base}/AP'] = float(metrics['ap'])
    elif task == "multi_label_classification":
      assert 'ap' in metrics, "Multi-label classification requires AP"
      wb_payload[f'{base}/AP'] = float(metrics['ap'])
    elif task == "multi_target_regression":
      assert 'mae' in metrics, "Multi-target regression requires MAE"
      wb_payload[f'{base}/MAE'] = float(metrics['mae'])
      if 'mse' in metrics:
        wb_payload[f'{base}/MSE'] = float(metrics['mse'])
      if 'r2' in metrics:
        wb_payload[f'{base}/R2'] = float(metrics['r2'])
    else:
        raise ValueError(f"Unsupported task type: {task}")
    wandb_logger.log(wb_payload)
