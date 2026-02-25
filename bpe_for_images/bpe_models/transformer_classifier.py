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

# 使用本子项目轻量实现
from bpe_for_images.models.encoder import create_encoder
from bpe_for_images.utils.logger import get_logger

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
        
        # 特殊token基准与有效词表大小（灰度/BPE + 8个特殊token）
        self._base_special_id = vocab_size
        self._effective_vocab_size = vocab_size + 8
        
        # 注意：VocabManager的vocab_size只包含特殊token（8个）
        # 但BERT的embedding需要支持灰度值token（0-255）+ 特殊token
        # 所以实际的vocab_size = vocab_size（灰度值256或BPE扩展后）+ VocabManager的特殊token数
        # 但VocabManager会自动管理，我们只需要确保BERT配置正确
        # VocabManager.vocab_size会在build_vocab后更新，但这里我们不需要build_vocab
        # 对于图像数据，我们直接将灰度值（0-255）映射到embedding层
        # 特殊token使用VocabManager管理的ID（0-7）
        # 所以实际BERT的vocab_size应该是：max(灰度值) + 1 + 特殊token数
        # 但为了简单，我们使用vocab_size（256或BPE扩展）+ 特殊token数（8）
        
        # 计算实际的BERT vocab_size
        # 灰度值token: 0-255 (256个)，或BPE扩展后的vocab_size
        # 特殊token: VocabManager管理的8个（ID: vocab_size 到 vocab_size+7）
        # 所以BERT的vocab_size需要支持 0 到 vocab_size+7
        # VocabManager.vocab_size会在创建后设置，这里先不设置
        # self.bert_vocab_size = vocab_size + 8  # 灰度值/BPE + 8个特殊token
        
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
    
    def _cls_token_id(self) -> int:
        return self._base_special_id + 3
    
    def _create_bert_encoder(self, config: Dict[str, Any]):
        """创建BERT编码器"""
        # 构造BERT配置（使用主项目的配置格式）
        # 注意：vocab_size需要包含特殊token，但VocabManager会自动管理
        # 对于图像数据，实际的embedding需要支持0-255灰度值 + 特殊token
        
        # VocabManager的vocab_size是从特殊token预留的，实际还需要加上灰度值token
        # 但我们从VocabManager获取vocab_size，然后加上灰度值token的数量
        # 简单起见，我们使用传入的vocab_size（256灰度值）+ 特殊token数量
        
        # 计算有效 vocab 与 pad
        base_special_id = self._base_special_id
        effective_vocab_size = self._effective_vocab_size

        # 严格参数传递（无默认值）
        encoder = create_encoder(
            'bert',
            vocab_size=effective_vocab_size,
            pad_token_id=base_special_id,
            hidden_size=config['d_model'],
            num_hidden_layers=config['n_layers'],
            num_attention_heads=config['n_heads'],
            intermediate_size=config['d_ff'],
            hidden_dropout_prob=config['dropout'],
            attention_probs_dropout_prob=config['attention_dropout'],
            max_position_embeddings=config['max_seq_length'],
            layer_norm_eps=config['layer_norm_eps'],
            type_vocab_size=2,
            initializer_range=config['initializer_range'],
            reset_weights=True,
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
        cls_token_id = self._cls_token_id()
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

