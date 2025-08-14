from __future__ import annotations

from typing import Dict, List, Any, Optional, TYPE_CHECKING

from config import ProjectConfig
from src.training.pretrain_pipeline import train_bert_mlm
from src.models.bert.vocab_manager import VocabManager, build_vocab_from_sequences

# if TYPE_CHECKING:
from src.data.unified_data_interface import UnifiedDataInterface


def build_vocab_from_tokens(token_sequences: Dict[str, List[List[int]]], config: ProjectConfig) -> VocabManager:
    """
    基于传入的 tokens（必须包含 'train'/'val'/'test' 三个划分）构建词表。
    不做任何数据加载/构建。
    """
    for key in ("train", "val", "test"):
        if key not in token_sequences:
            raise ValueError(f"缺少划分: {key}")

    all_sequences: List[List[int]] = (
        list(token_sequences["train"]) + list(token_sequences["val"]) + list(token_sequences["test"])
    )
    return build_vocab_from_sequences(all_sequences, config=config,
                                      min_freq=config.bert.pretraining.vocab_min_freq,
                                      max_vocab_size=config.bert.pretraining.max_vocab_size)


def pretrain(
    config: ProjectConfig,
    token_sequences: Dict[str, List[List[int]]],
    vocab_manager: VocabManager,
    udi: UnifiedDataInterface,
    method: str,
) -> Dict[str, Any]:
    """
    预训练入口（纯训练层）：
    - 消费调用方传入的 tokens（要求包含 train/val/test）与词表；
    - 不做任何数据加载/构建；
    - 支持BPE Transform（需要提供udi和method参数）；
    - 返回 {mlm_model, vocab_manager, stats}。
    """
    # 直接使用重构后的预训练函数
    return train_bert_mlm(config, token_sequences, vocab_manager, udi, method)





