from __future__ import annotations

from typing import Optional

import torch


def pool_sequence(
    sequence_output: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    method: str = "mean",
) -> torch.Tensor:
    """Sequence pooling function.
    公共序列池化函数。

    Args:
        sequence_output: [batch, seq_len, hidden]
        attention_mask: [batch, seq_len], 1=valid, 0=pad; can be None
                        1=有效，0=pad；可为 None
        method: 'mean' | 'cls'

    Returns:
        [batch, hidden]
    """
    if method == "cls":
        # Take the first token (data side must ensure it is the CLS/aggregate position)
        # 直接取首位（需由数据侧保证首位是CLS/聚合位）
        return sequence_output[:, 0, :]

    # Default: mean pooling (ignoring padding)
    # 默认 mean pooling（忽略padding）
    if attention_mask is None:
        # Degenerate to simple average / 退化为纯平均
        lengths = torch.full(
            (sequence_output.size(0), 1),
            fill_value=sequence_output.size(1),
            device=sequence_output.device,
            dtype=sequence_output.dtype,
        )
        return sequence_output.sum(dim=1) / torch.clamp(lengths, min=1e-9)

    if attention_mask.dtype != torch.bool:
        mask = attention_mask.to(dtype=torch.bool)
    else:
        mask = attention_mask

    # Expand mask to hidden dim for element-wise multiplication
    # 将 mask 扩展到 hidden 维进行逐元素相乘
    masked_output = sequence_output * mask.unsqueeze(-1).to(sequence_output.dtype)
    lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(sequence_output.dtype)
    return masked_output.sum(dim=1) / lengths


