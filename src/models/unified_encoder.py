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


from src.models.bert.vocab_manager import VocabManager
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

    def __init__(self, model_name: str, config: Dict[str, Any], vocab_manager: VocabManager):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.vocab_manager = vocab_manager

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
        """获取最大序列长度 - 兼容字典和HuggingFace配置对象"""
        # 支持字典式config (BERT自定义配置)
        if hasattr(self.config, 'get'):
            return int(self.config.get('max_position_embeddings', self.config.get('max_seq_length', 512)))
        # 支持HuggingFace config对象 (GTE等)
        else:
            return int(getattr(self.config, 'max_position_embeddings', getattr(self.config, 'max_seq_length', 512)))
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

    def __init__(self, model_name: str, config: Dict[str, Any], vocab_manager: VocabManager):
        super().__init__(model_name, config, vocab_manager)
        
        # 创建BERT配置（与原create_bert_mlm逻辑一致）
        bert_config = BertConfig(
            vocab_size=vocab_manager.vocab_size,
            hidden_size=config.get('hidden_size', 512),
            num_hidden_layers=config.get('num_hidden_layers', 4),
            num_attention_heads=config.get('num_attention_heads', 8),
            intermediate_size=config.get('intermediate_size', 2048),
            hidden_dropout_prob=config.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=config.get('attention_probs_dropout_prob', 0.1),
            max_position_embeddings=config.get('max_position_embeddings', 512),
            layer_norm_eps=config.get('layer_norm_eps', 1e-12),
            pad_token_id=vocab_manager.pad_token_id,
            dropout=config.get('hidden_dropout_prob', 0.1),
            type_vocab_size=config.get('type_vocab_size', 2),
            initializer_range=config.get('initializer_range', 0.02)
        )
        
        # 创建HuggingFace BERT模型（与原BertMLM逻辑一致）
        hf_config = bert_config.to_hf_config()
        self.bert = BertModel(hf_config)  # 直接创建，不需要BertMLM包装
        
        # 保存配置和词表管理器
        self.bert_config = bert_config
        self._hidden_size = config['hidden_size']

        # 统一从config读取reset标志
        reset_weights = bool(config['reset_weights'])
        if reset_weights:
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


class GTEEncoder(BaseEncoder):
    """GTE编码器适配器（Alibaba-NLP/gte-multilingual-base）。"""

    def __init__(self, model_name: str, config: Dict[str, Any], vocab_manager: VocabManager):
        super().__init__(model_name, config, vocab_manager)

        optimization = config['optimization']
        reset_weights = config['reset_weights']

        torch_dtype = parse_torch_dtype(optimization['torch_dtype'])

        # 词表与 pad 协议目标值（来自项目词表）
        target_vocab_size = int(vocab_manager.vocab_size)
        target_pad_id = int(vocab_manager.pad_token_id)

        if reset_weights:
            # 方案A：从配置新建模型，按库规则随机初始化全部参数
            logger.warning("🆕 使用AutoConfig.from_pretrained + AutoModel.from_config进行随机初始化GTE模型（丢弃预训练）")
            cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            # 统一写入优化与词表/Pad设定（先占位，稍后根据vocab_manager覆盖）
            cfg.unpad_inputs = optimization['unpad_inputs']
            cfg.use_memory_efficient_attention = optimization['use_memory_efficient_attention']
            cfg.torch_dtype = torch_dtype
            # 直接在配置阶段设置目标 vocab/pad，避免后续再 resize 带来的再初始化差异
            cfg.vocab_size = target_vocab_size
            cfg.pad_token_id = target_pad_id
            self.gte_model = AutoModel.from_config(cfg, torch_dtype=torch_dtype, trust_remote_code=True)
        else:
            logger.info("🔄 加载官方GTE预训练权重")
            self.gte_model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                unpad_inputs=optimization['unpad_inputs'],
                use_memory_efficient_attention=optimization['use_memory_efficient_attention'],
                torch_dtype=torch_dtype,
            )
        
        # 词表与 pad 协议对齐
        if reset_weights:
            # 已在配置阶段设置 vocab/pad，确保初始嵌入即为目标形状
            assert self.gte_model.config.vocab_size == target_vocab_size
            assert self.gte_model.config.pad_token_id == target_pad_id
            emb = self.gte_model.get_input_embeddings()
            emb.padding_idx = target_pad_id
            with torch.no_grad():
                emb.weight[target_pad_id].zero_()
            logger.info(f"✅ 随机初始化GTE模型并适配词表/Pad：vocab={target_vocab_size}, pad_id={target_pad_id}")
        else:
            # 预训练路径：先设置 pad_token_id，再调整嵌入尺寸，最后清零 pad 行
            self.gte_model.config.pad_token_id = target_pad_id
            self.gte_model.resize_token_embeddings(target_vocab_size)
            emb = self.gte_model.get_input_embeddings()
            emb.padding_idx = target_pad_id
            with torch.no_grad():
                emb.weight[target_pad_id].zero_()
            logger.info("📚 使用官方GTE预训练权重，仅适配词表")

        self._hidden_size = self.gte_model.config.hidden_size
        assert isinstance(self._hidden_size, int), "GTE编码器hidden_size必须是整数"
        # 使用底层 config，保持单一数据源
        self.config = self.gte_model.config

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


def create_encoder(model_name: str, config: Dict[str, Any], vocab_manager: VocabManager) -> BaseEncoder:
    name = (model_name or '').lower()
    if 'gte' in name:
      #note： 这个是必要的，因为如果不用本地这个目录的话，他会去尝试访问huggingface的。模型接口那个需要联网，而在服务器上会卡死。
        return GTEEncoder('./gte_model', config, vocab_manager)
    # 默认走bert
    return BertEncoder(model_name or 'bert', config, vocab_manager)


def list_supported_encoders() -> Dict[str, str]:
    return {
        'bert': 'Internal BERT implementation',
        'Alibaba-NLP/gte-multilingual-base': 'Alibaba GTE multilingual model',
    }


