from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class VariantWeightingAggregator(nn.Module):
    """Weighted aggregator for multi-variant features of the same graph.
    对同一图的多变体特征进行加权的聚合器。

    Input / 输入: features [B, K, D], mask [B, K] (bool)
    Output / 输出: weights [B, K] (softmax over valid variants; pad positions get weight 0)
                  （在有效变体上做softmax，pad位置权重为0）

    Note / 说明:
      - use_pred_as_feat: if True, normalized predictions are appended to features during training;
        evaluate_model handles the concatenation at inference time.
        若为 True，训练时应将标准化预测值附加到特征末尾；推理端 evaluate_model 中会根据该标志拼接。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1,
                 use_pred_as_feat: bool = False):
        super().__init__()
        self.use_pred_as_feat: bool = bool(use_pred_as_feat)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # per-variant score
        )

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Masked softmax: positions where mask=False are set to -inf before softmax.
        在mask指定的维度上做softmax。mask为False的位置置为-inf，softmax后近似0。"""
        # scores: [B, K, 1] or [B, K]
        if scores.dim() == 3 and scores.size(-1) == 1:
            scores = scores.squeeze(-1)
        # mask: [B, K] bool
        very_neg = torch.finfo(scores.dtype).min if scores.is_floating_point() else -1e9
        masked_scores = scores.masked_fill(~mask, very_neg)
        weights = torch.softmax(masked_scores, dim=dim)
        # Force pad positions to 0 (numerical stability) / 将pad位置强制为0（数值稳定）
        weights = weights * mask.float()
        # Renormalize to ensure valid weights sum to 1 (assumes at least 1 valid per graph)
        # 归一化以确保有效位置权重和为1（假设每图至少1个有效）
        denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return weights / denom

    def forward(self, features: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
          features: [B, K, D]
          mask: [B, K] bool, True = valid variant / True表示有效变体
        Returns:
          weights: [B, K]
        """
        assert features.dim() == 3, f"features must be [B, K, D], got {features.shape}"
        B, K, D = features.shape
        if mask is None:
            mask = torch.ones(B, K, dtype=torch.bool, device=features.device)
        scores = self.mlp(features)  # [B, K, 1]
        weights = self._masked_softmax(scores, mask=mask, dim=1)  # [B, K]
        return weights


