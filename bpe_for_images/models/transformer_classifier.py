"""
Transformer分类器
=================

基于主项目的BERT/GTE编码器实现的图像序列分类器
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 添加主项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

# 复用主项目的Transformer实现
from src.models.bert.vocab_manager import VocabManager
from src.models.unified_encoder import BertEncoder
from src.models.bert.config import BertConfig as MainBertConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerClassifier(nn.Module):
    """
    Transformer图像分类器
    
    架构:
    Input sequence -> [CLS] + tokens -> Transformer Encoder 
    -> Pooling (CLS/mean) -> Classification Head
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        transformer_config: Dict[str, Any],
        transformer_type: str = "bert",
        pooling_method: str = "cls"
    ):
        """
        Args:
            vocab_size: 词汇表大小（灰度值0-255，或BPE扩展后）
            num_classes: 分类类别数
            transformer_config: Transformer配置字典
            transformer_type: Transformer类型（"bert"或"gte"）
            pooling_method: 池化方法（"cls", "mean", "max"）
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.transformer_type = transformer_type
        self.pooling_method = pooling_method
        
        # 创建词汇表管理器（用于特殊token）
        self.vocab_manager = self._create_vocab_manager(vocab_size)
        
        # 创建Transformer编码器
        if transformer_type == "bert":
            self.encoder = self._create_bert_encoder(transformer_config)
        elif transformer_type == "gte":
            # GTE暂时使用BERT架构（因为我们从头训练）
            logger.warning("GTE类型暂时使用BERT架构实现（随机初始化）")
            self.encoder = self._create_bert_encoder(transformer_config)
        else:
            raise ValueError(f"不支持的transformer类型: {transformer_type}")
        
        # 分类头
        hidden_size = self.encoder.get_hidden_size()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        logger.info(f"创建TransformerClassifier: type={transformer_type}, "
                   f"vocab={vocab_size}, hidden={hidden_size}, classes={num_classes}")
    
    def _create_vocab_manager(self, vocab_size: int) -> VocabManager:
        """创建词汇表管理器"""
        # 定义特殊token
        special_tokens = {
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]',
            'unk_token': '[UNK]'
        }
        
        # 创建token到ID的映射
        token_to_id = {}
        id_to_token = {}
        
        # 先添加特殊token
        for idx, (key, token) in enumerate(special_tokens.items()):
            token_to_id[token] = idx
            id_to_token[idx] = token
        
        # 添加灰度值token（0-255）或BPE扩展token
        num_special = len(special_tokens)
        for i in range(vocab_size - num_special):
            token = str(i)  # 简单使用数字字符串作为token
            token_to_id[token] = num_special + i
            id_to_token[num_special + i] = token
        
        vocab_manager = VocabManager(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            special_tokens=special_tokens
        )
        
        return vocab_manager
    
    def _create_bert_encoder(self, config: Dict[str, Any]) -> BertEncoder:
        """创建BERT编码器"""
        # 构造BERT配置（使用主项目的配置格式）
        bert_config_dict = {
            'vocab_size': self.vocab_size,
            'hidden_size': config.get('d_model', 256),
            'num_hidden_layers': config.get('n_layers', 4),
            'num_attention_heads': config.get('n_heads', 4),
            'intermediate_size': config.get('d_ff', 1024),
            'max_position_embeddings': config.get('max_seq_length', 1024),
            'hidden_dropout_prob': config.get('dropout', 0.1),
            'attention_probs_dropout_prob': config.get('attention_dropout', 0.1),
            'hidden_act': config.get('activation', 'gelu'),
            'layer_norm_eps': config.get('layer_norm_eps', 1e-12),
            'initializer_range': config.get('initializer_range', 0.02),
        }
        
        # 创建编码器
        encoder = BertEncoder(
            model_name='bert_image',
            config=bert_config_dict,
            vocab_manager=self.vocab_manager
        )
        
        return encoder
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) 注意力掩码
        
        Returns:
            logits: (batch_size, num_classes)
        """
        # 添加[CLS] token
        batch_size = input_ids.size(0)
        cls_token_id = self.vocab_manager.get_token_id('[CLS]')
        cls_tokens = torch.full((batch_size, 1), cls_token_id, 
                               dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([cls_tokens, input_ids], dim=1)
        
        # 更新attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = torch.cat([
                torch.ones(batch_size, 1, device=attention_mask.device),
                attention_mask
            ], dim=1)
        
        # Transformer编码
        # 使用encode方法进行池化
        pooled_output = self.encoder.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_method=self.pooling_method
        )
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits
    
    def count_parameters(self) -> int:
        """统计参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_encoder_parameters(self) -> int:
        """统计编码器参数数量"""
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
    
    def get_classifier_parameters(self) -> int:
        """统计分类头参数数量"""
        return sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import BERT_CONFIG, GRAYSCALE_VOCAB_SIZE, NUM_CLASSES
    
    print("测试Transformer分类器...")
    
    # 创建模型
    model = TransformerClassifier(
        vocab_size=GRAYSCALE_VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        transformer_config=BERT_CONFIG,
        transformer_type="bert",
        pooling_method="cls"
    )
    
    print(f"\n模型架构摘要:")
    print(f"  总参数: {model.count_parameters():,}")
    print(f"  编码器参数: {model.get_encoder_parameters():,}")
    print(f"  分类头参数: {model.get_classifier_parameters():,}")
    
    # 测试前向传播
    batch_size = 8
    seq_len = 784  # MNIST展平长度
    
    # 创建随机输入（灰度值0-255）
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    
    # 测试无attention_mask
    print(f"\n测试前向传播（无mask）:")
    out1 = model(input_ids)
    print(f"  输入: {input_ids.shape} -> 输出: {out1.shape}")
    
    # 测试带attention_mask
    print(f"\n测试前向传播（带mask）:")
    attention_mask = torch.ones(batch_size, seq_len)
    out2 = model(input_ids, attention_mask)
    print(f"  输入: {input_ids.shape} + mask -> 输出: {out2.shape}")
    
    # 测试梯度
    print(f"\n测试梯度反向传播:")
    loss = out2.sum()
    loss.backward()
    print(f"  反向传播成功")
    
    print("\n测试通过！")

