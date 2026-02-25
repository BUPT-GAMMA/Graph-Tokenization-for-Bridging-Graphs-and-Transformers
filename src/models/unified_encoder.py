"""
统一编码器模型工厂（恢复）
=========================

提供统一的编码器模型接口，支持多种预训练模型：
- BERT（项目内置）
- GTE（Alibaba-NLP/gte-multilingual-base）

设计要点：
- 抽象接口 BaseEncoder：输入 token ids / attention_mask，输出句向量
- 统一任务包装器 UnifiedTaskModel：encoder.encode -> 任务头 -> 输出
- 工厂方法 create_encoder：最小改动接入其他encoder
"""

from __future__ import annotations
from typing import Dict, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


from src.models.utils.pooling import pool_sequence
from src.utils.check import parse_torch_dtype
from src.utils.logger import get_logger

# 模型相关导入
from transformers import BertModel, AutoModel, AutoConfig
from src.models.bert.config import BertConfig

# 创建模块级logger
logger = get_logger(__name__)


class BaseEncoder(nn.Module, ABC):
    """编码器基类：定义统一API（sequence -> vector）。"""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    @abstractmethod
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, pooling_method: str = 'mean') -> torch.Tensor:
        """编码并池化，返回句子级表示 [batch, hidden]."""
        pass
    
    @abstractmethod
    def get_sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """获取序列级编码输出，不进行池化 [batch, seq_len, hidden] - MLM任务需要"""
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        pass

    def get_max_seq_length(self) -> int:
        """获取最大序列长度（由子类实现）。"""
        raise NotImplementedError
    @abstractmethod
    def get_word_embeddings_weight(self) -> torch.nn.Parameter:
        pass
    def save_model(self, save_path: str) -> None:
        # 默认无特殊保存逻辑，由具体实现决定
        pass

    @classmethod
    def load_model(cls, model_path: str, model_name: str, config: Dict[str, Any]) -> 'BaseEncoder':
        raise NotImplementedError


class BertEncoder(BaseEncoder):
    """BERT编码器适配器（基于项目内置BERT）。"""

    def __init__(self,
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
                 reset_weights: bool):
        super().__init__(model_name)

        # 创建BERT配置（严格使用传入参数，不做fallback）
        bert_config = BertConfig(
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
            dropout=float(hidden_dropout_prob),
            type_vocab_size=int(type_vocab_size),
            initializer_range=float(initializer_range)
        )

        # 创建HuggingFace BERT模型
        hf_config = bert_config.to_hf_config()
        self.bert = BertModel(hf_config)

        self.bert_config = bert_config
        self._hidden_size = int(hidden_size)
        self._max_position_embeddings = int(max_position_embeddings)

        if bool(reset_weights):
            self._reinitialize_bert_weights()
    
    def get_word_embeddings_weight(self) -> torch.nn.Parameter:
        return self.bert.embeddings.word_embeddings.weight  # [V, H]
      
    def _reinitialize_bert_weights(self):
        """重新初始化BERT权重"""
        logger.info("🔄 重新初始化BERT权重...")
        for module in self.bert.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    torch.nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        logger.info("✅ BERT权重重新初始化完成")

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, pooling_method: str = 'mean') -> torch.Tensor:
        """句子级编码 - 获取池化后的表示 [batch, hidden_size]"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        if pooling_method == 'pooler':
            raise ValueError("BertEncoder 不支持 'pooler' 池化；请使用 'mean'、'max' 或 'cls'")
        return pool_sequence(sequence_output, attention_mask, method=pooling_method)
    
    def get_sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """获取BERT序列级编码输出 [batch, seq_len, hidden_size] - MLM任务使用"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_max_seq_length(self) -> int:
        return self._max_position_embeddings


class GTEEncoder(BaseEncoder):
    """GTE编码器适配器（Alibaba-NLP/gte-multilingual-base）。"""

    def __init__(self,
                 model_name: str,
                 *,
                 vocab_size: int,
                 pad_token_id: int,
                 reset_weights: bool,
                 torch_dtype: str,
                 unpad_inputs: bool,
                 use_memory_efficient_attention: bool):
        super().__init__(model_name)

        dtype = parse_torch_dtype(torch_dtype)

        target_vocab_size = int(vocab_size)
        target_pad_id = int(pad_token_id)

        if reset_weights:
            logger.warning("🆕 使用AutoConfig.from_pretrained + AutoModel.from_config进行随机初始化GTE模型（丢弃预训练）")
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            cfg.unpad_inputs = bool(unpad_inputs)
            cfg.use_memory_efficient_attention = bool(use_memory_efficient_attention)
            cfg.torch_dtype = dtype
            cfg.vocab_size = target_vocab_size
            cfg.pad_token_id = target_pad_id
            self.gte_model = AutoModel.from_config(cfg, torch_dtype=dtype, trust_remote_code=True)
        else:
            logger.info("🔄 加载官方GTE预训练权重")
            self.gte_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                unpad_inputs=bool(unpad_inputs),
                use_memory_efficient_attention=bool(use_memory_efficient_attention),
                torch_dtype=dtype,
            )

        # 词表与 pad 协议对齐
        if reset_weights:
            assert self.gte_model.config.vocab_size == target_vocab_size
            assert self.gte_model.config.pad_token_id == target_pad_id
            emb = self.gte_model.get_input_embeddings()
            emb.padding_idx = target_pad_id
            with torch.no_grad():
                emb.weight[target_pad_id].zero_()
            logger.info(f"✅ 随机初始化GTE模型并适配词表/Pad：vocab={target_vocab_size}, pad_id={target_pad_id}")
        else:
            self.gte_model.config.pad_token_id = target_pad_id
            self.gte_model.resize_token_embeddings(target_vocab_size)
            emb = self.gte_model.get_input_embeddings()
            emb.padding_idx = target_pad_id
            with torch.no_grad():
                emb.weight[target_pad_id].zero_()
            logger.info("📚 使用官方GTE预训练权重，仅适配词表")

        self._hidden_size = int(self.gte_model.config.hidden_size)
        self._max_position_embeddings = int(getattr(self.gte_model.config, 'max_position_embeddings', 512))

    def get_word_embeddings_weight(self) -> torch.nn.Parameter:
        return self.gte_model.get_input_embeddings().weight  # [V, H]

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, pooling_method: str = 'mean') -> torch.Tensor:
        outputs = self.gte_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if pooling_method == 'pooler':
            return outputs.pooler_output
        sequence_output = outputs.last_hidden_state
        return pool_sequence(sequence_output, attention_mask, method=('cls' if pooling_method == 'cls' else 'mean'))
    
    def get_sequence_output(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """获取GTE序列级编码输出 [batch, seq_len, hidden_size] - MLM任务使用"""
        outputs = self.gte_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

    def get_hidden_size(self) -> int:
        return self._hidden_size

    def get_max_seq_length(self) -> int:
        return self._max_position_embeddings


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
    torch_dtype: str,
    unpad_inputs: bool,
    use_memory_efficient_attention: bool,
) -> BaseEncoder:
    name = (model_name or '').lower()
    if 'gte' in name:
        return GTEEncoder(
            './gte_model',
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            reset_weights=reset_weights,
            torch_dtype=torch_dtype,
            unpad_inputs=unpad_inputs,
            use_memory_efficient_attention=use_memory_efficient_attention,
        )
    return BertEncoder(
        model_name or 'bert',
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


def create_encoder_from_config(model_name: str, config: Dict[str, Any], vocab_manager) -> BaseEncoder:
    """便利接口：从 config 与 vocab_manager 中提取参数创建编码器。
    - 对于 'gte'：严格读取 optimization 字段；
    - 对于 'bert'：不读取 optimization 字段。
    """
    name = (model_name or '').lower()
    vocab_size = int(vocab_manager.vocab_size)
    pad_token_id = int(vocab_manager.pad_token_id)

    if 'gte' in name:
        opt = config['optimization']
        return GTEEncoder(
            './gte_model',
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            reset_weights=config['reset_weights'],
            torch_dtype=str(opt['torch_dtype']),
            unpad_inputs=bool(opt['unpad_inputs']),
            use_memory_efficient_attention=bool(opt['use_memory_efficient_attention']),
        )

    return create_encoder(
        model_name,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        hidden_dropout_prob=config['hidden_dropout_prob'],
        attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
        max_position_embeddings=config['max_position_embeddings'],
        layer_norm_eps=config['layer_norm_eps'],
        type_vocab_size=config['type_vocab_size'],
        initializer_range=config['initializer_range'],
        reset_weights=config['reset_weights'],
        torch_dtype='float32',
        unpad_inputs=False,
        use_memory_efficient_attention=False,
    )


def list_supported_encoders() -> Dict[str, str]:
    return {
        'bert': 'Internal BERT implementation',
        'Alibaba-NLP/gte-multilingual-base': 'Alibaba GTE multilingual model',
    }


