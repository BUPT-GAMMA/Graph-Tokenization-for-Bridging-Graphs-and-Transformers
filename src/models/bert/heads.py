from __future__ import annotations

from typing import Literal, Optional

from src.models.bert.model import (
    BertRegression,
    BertClassification,
    BertConfig,
)



# todo 注意这个地方写的完全不对。既然名称是create task head，那么他创建的自然应该只是一个head。然后把这个孩子可能在上层用来接在整个bert微调的模型里面去。但是现在它实际上是创建了一整个bert微调的模型，而它的参数则是一从完整的预训练模型中取得的。这不对。
def create_task_head(
    task: Literal["regression", "classification"],
    *,
    vocab_manager,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    pooling_method: str,
    dropout: float,
    max_position_embeddings: int,
    layer_norm_eps: float,
    num_classes: Optional[int] = None,
):
    if task == "regression":
        cfg = BertConfig(
            vocab_size=vocab_manager.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=vocab_manager.pad_token_id,
        )
        return BertRegression(
            config=cfg,
            vocab_manager=vocab_manager,
            pooling_method=pooling_method,
        )
    elif task == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification task head")
        cfg = BertConfig(
            vocab_size=vocab_manager.vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=vocab_manager.pad_token_id,
        )
        return BertClassification(
            config=cfg,
            vocab_manager=vocab_manager,
            num_classes=num_classes,
            pooling_method=pooling_method,
        )
    else:
        raise ValueError(f"Unsupported task head: {task}")


