from __future__ import annotations

from typing import Optional, Literal


def load_pretrained_backbone(config):
    """加载预训练 BERT（MLM）作为微调 backbone。"""
    from src.models.bert.model import BertMLM
    pretrained_dir = config.get_bert_model_path("pretrained").parent
    return BertMLM.load_model(str(pretrained_dir))


def build_task_model(
    config,
    task: Literal["regression", "classification"],
    *,
    pretrained,
    num_classes: Optional[int] = None,
):
    """在 backbone 上构建任务模型，并加载 backbone 权重。"""
    from src.models.bert.heads import create_task_head

    pooling = config.bert.architecture.pooling_method
    dropout = 0.2
    model = create_task_head(
        task=task,
        vocab_manager=pretrained.vocab_manager,
        hidden_size=pretrained.config.hidden_size,
        num_hidden_layers=pretrained.config.num_hidden_layers,
        num_attention_heads=pretrained.config.num_attention_heads,
        intermediate_size=pretrained.config.intermediate_size,
        pooling_method=pooling,
        dropout=dropout,
        max_position_embeddings=pretrained.config.max_position_embeddings,
        layer_norm_eps=pretrained.config.layer_norm_eps,
        num_classes=num_classes,
    )
    model.bert.load_state_dict(pretrained.bert.state_dict())
    return model





