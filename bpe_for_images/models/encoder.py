"""
Lightweight encoder utilities for BPE-Images project
Independent from tokenizerGraph src tree.
"""

from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BertEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_token_id: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        max_position_embeddings: int,
        layer_norm_eps: float,
        type_vocab_size: int,
        initializer_range: float,
        reset_weights: bool,
    ) -> None:
        super().__init__()
        cfg = BertConfig(
            vocab_size=int(vocab_size),
            hidden_size=int(hidden_size),
            num_hidden_layers=int(num_hidden_layers),
            num_attention_heads=int(num_attention_heads),
            intermediate_size=int(intermediate_size),
            hidden_dropout_prob=float(hidden_dropout_prob),
            attention_probs_dropout_prob=float(attention_probs_dropout_prob),
            max_position_embeddings=int(max_position_embeddings),
            layer_norm_eps=float(layer_norm_eps),
            pad_token_id=int(pad_token_id),
            type_vocab_size=int(type_vocab_size),
            initializer_range=float(initializer_range),
        )
        self.bert = BertModel(cfg)
        self._hidden_size = int(hidden_size)
        self._max_position_embeddings = int(max_position_embeddings)
        if reset_weights:
            self._reset_weights()

    def _reset_weights(self) -> None:
        for module in self.bert.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, pooling_method: str = 'cls') -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        seq = out.last_hidden_state
        if pooling_method == 'cls':
            return seq[:, 0]
        elif pooling_method == 'mean':
            if attention_mask is None:
                return seq.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            return (seq * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        elif pooling_method == 'max':
            return seq.max(dim=1).values
        else:
            raise ValueError(f"Unsupported pooling: {pooling_method}")

    def get_sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_max_seq_length(self) -> int:
        return self._max_position_embeddings

    def get_word_embeddings_weight(self) -> torch.nn.Parameter:
        return self.bert.embeddings.word_embeddings.weight


def create_encoder(
    model_name: str,
    *,
    vocab_size: int,
    pad_token_id: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_position_embeddings: int,
    layer_norm_eps: float,
    type_vocab_size: int,
    initializer_range: float,
    reset_weights: bool,
) -> BertEncoder:
    name = (model_name or '').lower()
    if name != 'bert' and 'bert' not in name:
        raise ValueError('Only bert is supported in bpe_for_images')
    return BertEncoder(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        type_vocab_size=type_vocab_size,
        initializer_range=initializer_range,
        reset_weights=reset_weights,
    )










