from __future__ import annotations

from typing import Dict, Any, Tuple
from pathlib import Path
import torch


def update_and_check(
    best_metric: float,
    new_metric: float,
    patience_counter: int,
    patience: int,
) -> Tuple[float, int, bool]:
    """
    更新 best 与 patience 计数器；返回 (best_metric, patience_counter, should_stop)
    """
    if new_metric < best_metric:
        return new_metric, 0, False
    patience_counter += 1
    return best_metric, patience_counter, (patience_counter >= patience)


def save_best(model, path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        **meta,
    }, str(path))


def save_final(model, path: Path, meta: Dict[str, Any]) -> None:
    save_best(model, path, meta)



