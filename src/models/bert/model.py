"""
BERT Model Implementation for Token ID Sequences
支持Token ID序列输入的BERT模型实现
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from transformers import BertModel

from .vocab_manager import VocabManager
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface


class BertConfig:
    """BERT配置"""
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 512,
                 layer_norm_eps: float = 1e-12,
                 pad_token_id: int = 0,
                 dropout: float = 0.1,  # 增加dropout防止过拟合 
                 **kwargs):
        """
        初始化BERT配置
        
        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度，对应config.bert.architecture.hidden_size
            num_hidden_layers: BERT层数，对应config.bert.architecture.num_hidden_layers
            num_attention_heads: 注意力头数，对应config.bert.architecture.num_attention_heads
            intermediate_size: 前馈网络维度，对应config.bert.architecture.intermediate_size
            hidden_dropout_prob: 隐藏层dropout，对应config.bert.architecture.hidden_dropout_prob
            attention_probs_dropout_prob: 注意力dropout，对应config.bert.architecture.attention_probs_dropout_prob
            max_position_embeddings: 最大序列长度，对应config.bert.architecture.max_position_embeddings
            layer_norm_eps: 层归一化epsilon，对应config.bert.architecture.layer_norm_eps
            pad_token_id: padding token的ID，对应config.special_tokens.ids.pad
            dropout: dropout率，对应config.bert.architecture.dropout
            **kwargs: 其他配置参数
        """
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.dropout = dropout
        
        # 额外配置
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_hf_config(self):
        """转换为HuggingFace BertConfig"""
        from transformers import BertConfig as HFBertConfig
        return HFBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
            max_position_embeddings=self.max_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            position_embedding_type="absolute",
            use_cache=False
        )


class BertMLM(nn.Module):
    """BERT MLM模型 - 支持额外token特征融合"""
    
    def __init__(self, config: BertConfig, vocab_manager: VocabManager, feature_dim: Optional[int] = None):
        super().__init__()
        
        self.config: BertConfig = config
        self.vocab_manager: VocabManager = vocab_manager
        self.feature_dim = feature_dim
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # **关键修改**：只有在需要特征融合时才创建word_embeddings层
        if feature_dim is not None:
            self.word_embeddings = nn.Embedding(
                vocab_manager.vocab_size, 
                config.hidden_size, 
                padding_idx=hf_config.pad_token_id
            )
            # 特征投影层
            self.feature_projection = nn.Sequential(
                nn.Linear(feature_dim, config.hidden_size, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            )
        else:
            self.word_embeddings = None
            self.feature_projection = None
        
        # MLM头
        self.mlm_head = nn.Linear(config.hidden_size, vocab_manager.vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 只有在支持特征融合时才复制BERT的word embedding权重
        if self.feature_projection is not None:
            with torch.no_grad():
                self.word_embeddings.weight.copy_(self.bert.embeddings.word_embeddings.weight)
        
        # MLM头的权重初始化
        nn.init.normal_(self.mlm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)
        
        # 特征投影层初始化
        if self.feature_projection is not None:
            # 对Sequential中的每个Linear层进行初始化
            for module in self.feature_projection:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                token_features: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_features: [batch_size, seq_len, feature_dim] 可选的额外token特征
            labels: [batch_size, seq_len] MLM标签，-100表示不计算损失的位置
            
        Returns:
            包含logits和loss（如果提供labels）的字典
        """
        
        # **兼容性检查**：如果没有特征或不支持特征，使用原有流程
        if self.feature_projection is None :
            # 完全使用原有的BERT流程，确保向后兼容
            bert_outputs = self.bert(
                input_ids=input_ids,  # 使用原有的input_ids参数
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            # **特征融合流程**：只有在明确需要融合特征时才使用
            # 步骤1：手动计算word embedding
            word_embeds = self.word_embeddings(input_ids)  # [batch_size, seq_len, hidden_size]
            
            # 步骤2：投影并融合特征
            projected_features = self.feature_projection(token_features)  # [batch_size, seq_len, hidden_size]
            fused_embeds = word_embeds + projected_features
            
            # 步骤3：使用inputs_embeds参数
            """  note: 
            > if inputs_embeds is None:
            >     inputs_embeds = self.word_embeddings(input_ids)
            > token_type_embeddings = self.token_type_embeddings(token_type_ids)
            
            > self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

            """
            bert_outputs = self.bert(
                inputs_embeds=fused_embeds,  # 使用融合后的embedding
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # [batch_size, seq_len, hidden_size]
        sequence_output = bert_outputs.last_hidden_state
        
        # MLM预测 [batch_size, seq_len, vocab_size]
        mlm_logits = self.mlm_head(sequence_output)
        
        outputs: Dict[str, torch.Any] = {'logits': mlm_logits}
        
        # 计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            mlm_loss = loss_fct(mlm_logits.view(-1, self.vocab_manager.vocab_size), 
                               labels.view(-1))
            outputs['loss'] = mlm_loss
        
        return outputs
    
    def get_embeddings(self, input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor,
                      token_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取序列的embedding表示
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码
            token_features: [batch_size, seq_len, feature_dim] 可选的额外token特征
        
        Returns:
            [batch_size, seq_len, hidden_size] 的embedding
        """
        with torch.no_grad():
            # **兼容性检查**：如果没有特征或不支持特征，使用原有流程
            if self.feature_projection is None or token_features is None:
                # 完全使用原有的BERT流程，确保向后兼容
                bert_outputs = self.bert(
                    input_ids=input_ids,  # 使用原有的input_ids参数
                    attention_mask=attention_mask,
                    return_dict=True
                )
            else:
                # **特征融合流程**：使用与forward相同的逻辑
                word_embeds = self.word_embeddings(input_ids)
                projected_features = self.feature_projection(token_features)
                fused_embeds = word_embeds + projected_features
                
                bert_outputs = self.bert(
                    inputs_embeds=fused_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            return bert_outputs.last_hidden_state
    
    def save_model(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存配置（包含feature_dim）。不再保存词表，词表与数据集绑定，由 UDI 提供
        config_to_save = {'config': self.config, 'feature_dim': self.feature_dim}
        torch.save(config_to_save, os.path.join(save_path, 'config.bin'))
        
        print(f"模型已保存到: {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str, project_config: Optional['ProjectConfig'] = None) -> 'BertMLM':
        """加载模型
        
        Args:
            model_path: 模型路径
            project_config: 项目配置（用于加载词表），如果为None则尝试从环境推断
        """
        import os
        
        # 加载配置
        config_data = torch.load(os.path.join(model_path, 'config.bin'))
        
        # 兼容旧版本配置格式
        config = config_data['config']
        feature_dim = config_data['feature_dim']

        
        # 加载词表：词表与数据集绑定，从 UDI 读取（严格，不再从模型目录读取）
        if project_config is None:
            # 如果没有提供project_config，创建一个默认的
            from config import ProjectConfig
            project_config = ProjectConfig()
            # 从模型路径推断一些配置信息
            path_parts = model_path.split('/')
            # 尝试从路径中提取数据集名称
            for part in path_parts:
                if 'qm9' in part.lower():
                    project_config.dataset.name = part
                    break
            # 尝试从路径中提取序列化方法和BPE状态
            for part in path_parts:
                if '-' in part:
                    sub_parts = part.split('-')
                    for sub_part in sub_parts:
                        if sub_part.lower() == 'bpe':
                            # BPE在路径中仅为标识，具体配置由config决定
                            pass
                        elif sub_part in ['feuler', 'eulerian', 'dfs', 'bfs']:
                            project_config.serialization.method = sub_part
        
        udi = UnifiedDataInterface(project_config, project_config.dataset.name)
        # 从UDI获取完整词表（与数据集绑定，包含所有token类型）
        vocab_manager = udi.get_vocab(method=project_config.serialization.method)
        
        # 创建模型
        model = cls(config, vocab_manager, feature_dim)
        
        # 加载权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), 
                               map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"MLM模型已从 {model_path}/pytorch_model.bin 加载完成")
        return model


class BertRegression(nn.Module):
    """BERT回归模型 - 对整个序列预测一个连续数值"""
    
    def __init__(self, config: BertConfig, vocab_manager: VocabManager,
                 num_targets: int = 1, pooling_method: str = 'mean'):
        super().__init__()
        
        self.config = config
        self.vocab_manager = vocab_manager
        self.num_targets = num_targets
        self.pooling_method = pooling_method
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # 回归头：支持多目标输出
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, self.num_targets)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def _pool_sequence(self, sequence_output: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """对序列进行池化
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            [batch_size, hidden_size] 池化后的表示
        """
        if self.pooling_method == 'mean':
            # 平均池化，忽略padding位置
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling_method == 'cls':
            # 使用[CLS] token的表示
            return sequence_output[:, 0, :]
            
        elif self.pooling_method == 'max':
            # 最大池化
            sequence_output = sequence_output.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )
            return torch.max(sequence_output, dim=1)[0]
        
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码  
            labels: [batch_size, 1] 回归目标值
            
        Returns:
            包含predictions和loss（如果提供labels）的字典
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [batch_size, seq_len, hidden_size]
        sequence_output = bert_outputs.last_hidden_state
        
        # 池化: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        pooled_output = self._pool_sequence(sequence_output, attention_mask)

        # 回归预测: [batch_size, hidden_size] -> [batch_size, num_targets]
        predictions = self.regression_head(pooled_output)

        outputs: Dict[str, torch.Any] = {'predictions': predictions, 'pooled': pooled_output}
        
        # 计算损失
        if labels is not None:
            # 根据目标数量选择损失函数
            if self.num_targets == 1:
                # 单目标回归：使用MSE
                if labels.dim() == 1:
                    labels = labels.unsqueeze(-1)
                loss_fct = nn.MSELoss()
            else:
                # 多目标回归：使用MAE
                loss_fct = nn.L1Loss()
            
            regression_loss = loss_fct(predictions, labels.float())
            outputs['loss'] = regression_loss
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """预测连续数值"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['predictions']
    
    def save_model(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存配置（不保存词表）
        config_to_save = {'config': self.config, 'pooling_method': self.pooling_method}
        torch.save(config_to_save, os.path.join(save_path, 'config.bin'))
        
        print(f"回归模型已保存到: {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BertRegression':
        """加载模型"""
        import os
        
        # 加载配置
        config_data = torch.load(os.path.join(model_path, 'config.bin'))
        config = config_data['config']
        pooling_method = config_data['pooling_method']
        
        # 加载词表：词表与数据集绑定，从 UDI 读取
        cfg = ProjectConfig() if not isinstance(config, ProjectConfig) else config
        udi = UnifiedDataInterface(cfg, cfg.dataset.name)
        vocab_manager = udi.get_vocab(method=cfg.serialization.method)
        
        # 创建模型
        model = cls(config, vocab_manager, pooling_method)
        
        # 加载权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), 
                               map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"回归模型已从 {model_path} 加载完成")
        return model


class BertClassification(nn.Module):
    """BERT多标签分类模型 - 对整个序列预测多个二分类标签"""
    
    def __init__(self, config: BertConfig, vocab_manager: VocabManager,
                 num_labels: int, pooling_method: str = 'mean'):
        super().__init__()
        
        self.config = config
        self.vocab_manager = vocab_manager
        self.num_labels = num_labels
        self.pooling_method = pooling_method
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # 多标签分类头：[B, d] -> [B, num_labels]
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, num_labels)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def _pool_sequence(self, sequence_output: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """序列池化"""
        if self.pooling_method == 'mean':
            # 平均池化 (忽略padding)
            lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            masked_output = sequence_output * attention_mask.unsqueeze(-1)  # [batch_size, seq_len, hidden_size]
            pooled = masked_output.sum(dim=1) / lengths  # [batch_size, hidden_size]
            return pooled
        elif self.pooling_method == 'cls':
            # 使用[CLS] token的表示
            return sequence_output[:, 0, :]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码  
            labels: [batch_size, num_labels] 多标签（浮点数，0或1）
            
        Returns:
            包含logits和loss（如果提供labels）的字典
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 池化
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = self._pool_sequence(sequence_output, attention_mask)
        
        # 多标签分类预测: [batch_size, hidden_size] -> [batch_size, num_labels]
        logits = self.classification_head(pooled_output)
        
        outputs: Dict[str, torch.Any] = {'logits': logits, 'pooled': pooled_output}
        
        # 计算损失
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            outputs['loss'] = loss
        
        return outputs
    
    def predict_proba(self, input_ids: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """预测每个标签的概率"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return torch.sigmoid(outputs['logits'])


class BertMultiTargetRegression(nn.Module):
    """BERT多目标回归模型 - 对整个序列预测多个连续数值"""
    
    def __init__(self, config: BertConfig, vocab_manager: VocabManager,
                 num_targets: int, pooling_method: str = 'mean'):
        super().__init__()
        
        self.config = config
        self.vocab_manager = vocab_manager
        self.num_targets = num_targets
        self.pooling_method = pooling_method
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # 多目标回归头：[B, d] -> [B, num_targets]
        self.regression_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, num_targets)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.regression_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def _pool_sequence(self, sequence_output: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """序列池化"""
        if self.pooling_method == 'mean':
            # 平均池化 (忽略padding)
            lengths = attention_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            masked_output = sequence_output * attention_mask.unsqueeze(-1)  # [batch_size, seq_len, hidden_size]
            pooled = masked_output.sum(dim=1) / lengths  # [batch_size, hidden_size]
            return pooled
        elif self.pooling_method == 'cls':
            # 使用[CLS] token的表示
            return sequence_output[:, 0, :]  # [batch_size, hidden_size]
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码  
            labels: [batch_size, num_targets] 多目标回归值
            
        Returns:
            包含predictions和loss（如果提供labels）的字典
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 池化
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = self._pool_sequence(sequence_output, attention_mask)
        
        # 多目标回归预测: [batch_size, hidden_size] -> [batch_size, num_targets]
        predictions = self.regression_head(pooled_output)
        
        outputs: Dict[str, torch.Any] = {'predictions': predictions, 'pooled': pooled_output}
        
        # 计算损失
        if labels is not None:
            loss_fct = nn.L1Loss()  # 使用MAE损失
            loss = loss_fct(predictions, labels.float())
            outputs['loss'] = loss
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """预测多个回归目标"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['predictions']


class BertClassification(nn.Module):
    """BERT分类模型 - 支持单标签和多标签分类"""
    
    def __init__(self, config: BertConfig, vocab_manager: VocabManager,
                 num_classes: int, is_multi_label: bool = False, pooling_method: str = 'mean'):
        super().__init__()
        
        self.config = config
        self.vocab_manager = vocab_manager
        self.num_classes = num_classes
        self.is_multi_label = is_multi_label
        self.pooling_method = pooling_method
        
        # 更新配置中的词表大小
        self.config.vocab_size = vocab_manager.vocab_size
        
        # 创建HuggingFace BERT模型
        hf_config = config.to_hf_config()
        self.bert = BertModel(hf_config)
        
        # 分类头：[B, C, d] -> 池化 -> [B, d] -> MLP -> [B, num_classes]
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def _pool_sequence(self, sequence_output: torch.Tensor, 
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """对序列进行池化
        
        Args:
            sequence_output: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            [batch_size, hidden_size] 池化后的表示
        """
        if self.pooling_method == 'mean':
            # 平均池化，忽略padding位置
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.sum(mask_expanded, dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling_method == 'cls':
            # 使用[CLS] token的表示
            return sequence_output[:, 0, :]
            
        elif self.pooling_method == 'max':
            # 最大池化
            sequence_output = sequence_output.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )
            return torch.max(sequence_output, dim=1)[0]
        
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            input_ids: [batch_size, seq_len] token ID序列
            attention_mask: [batch_size, seq_len] 注意力掩码  
            labels: [batch_size] 分类标签 (整数)
            
        Returns:
            包含logits和loss（如果提供labels）的字典
        """
        # BERT编码
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [batch_size, seq_len, hidden_size]
        sequence_output = bert_outputs.last_hidden_state
        
        # 池化: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        pooled_output = self._pool_sequence(sequence_output, attention_mask)
        
        # 分类预测: [batch_size, hidden_size] -> [batch_size, num_classes]
        logits = self.classification_head(pooled_output)
        
        outputs: Dict[str, torch.Any] = {'logits': logits, 'pooled': pooled_output}
        
        # 计算损失
        if labels is not None:
            if self.is_multi_label:
                # 多标签分类：使用BCEWithLogitsLoss
                loss_fct = nn.BCEWithLogitsLoss()
                classification_loss = loss_fct(logits, labels.float())
            else:
                # 单标签分类：使用CrossEntropyLoss
                loss_fct = nn.CrossEntropyLoss()
                classification_loss = loss_fct(logits, labels.long())
            outputs['loss'] = classification_loss
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['logits']
    
    def predict_proba(self, input_ids: torch.Tensor, 
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """预测概率分布"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            if self.is_multi_label:
                # 多标签分类：每个标签独立的sigmoid概率
                return torch.sigmoid(logits)
            else:
                # 单标签分类：softmax概率分布
                return torch.softmax(logits, dim=-1)
    
    def predict_classes(self, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """预测类别标签"""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            return torch.argmax(logits, dim=-1)
    
    def save_model(self, save_path: str):
        """保存模型"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存配置（不保存词表）
        config_to_save = {'config': self.config, 'num_classes': self.num_classes, 'pooling_method': self.pooling_method}
        torch.save(config_to_save, os.path.join(save_path, 'config.bin'))
        
        print(f"分类模型已保存到: {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str) -> 'BertClassification':
        """加载模型"""
        import os
        
        # 加载配置
        config_data = torch.load(os.path.join(model_path, 'config.bin'))
        config = config_data['config']
        num_classes = config_data['num_classes']
        pooling_method = config_data.get('pooling_method', 'mean')
        
        # 加载词表：词表与数据集绑定，从 UDI 读取
        cfg = ProjectConfig() if not isinstance(config, ProjectConfig) else config
        udi = UnifiedDataInterface(cfg, cfg.dataset.name)
        vocab_manager = udi.get_vocab(method=cfg.serialization.method)
        
        # 创建模型
        model = cls(config, vocab_manager, num_classes, pooling_method)
        
        # 加载权重
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), 
                               map_location='cpu')
        model.load_state_dict(state_dict)
        
        print(f"分类模型已从 {model_path} 加载完成")
        return model


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """获取模型大小信息"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_param_count,
        'non_trainable_parameters': param_count - trainable_param_count
    }


def print_model_info(model: nn.Module , config: BertConfig):
    """打印模型信息"""
    size_info = get_model_size(model)
    
    print("=" * 50)
    print("BertMLM 模型信息")
    print("=" * 50)
    print(f"词汇表大小: {config.vocab_size:,}")
    print(f"模型维度: {config.hidden_size}")
    print(f"注意力头数: {config.num_attention_heads}")
    print(f"层数: {config.num_hidden_layers}")
    print(f"前馈网络维度: {config.intermediate_size}")
    print(f"最大序列长度: {config.max_position_embeddings}")
    print(f"dropout: {config.dropout}")
    print("-" * 50)
    print(f"总参数数量: {size_info['total_parameters']:,}")
    print(f"可训练参数: {size_info['trainable_parameters']:,}")
    print(f"不可训练参数: {size_info['non_trainable_parameters']:,}")
    print(f"模型大小: {size_info['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50) 


# 便捷的创建函数
def create_bert_mlm(vocab_manager: VocabManager,
                   hidden_size: int = 768,
                   num_hidden_layers: int = 12,
                   num_attention_heads: int = 12,
                   feature_dim: Optional[int] = None,
                   **kwargs) -> BertMLM:
    """创建BERT MLM模型
    
    Args:
        vocab_manager: 词表管理器
        hidden_size: BERT隐藏层维度
        num_hidden_layers: BERT层数
        num_attention_heads: 注意力头数
        feature_dim: 额外token特征的维度，如果为None则不支持特征融合
        **kwargs: 其他配置参数
    """
    config = BertConfig(
        vocab_size=vocab_manager.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        pad_token_id=vocab_manager.pad_token_id,
        **kwargs
    )
    
    return BertMLM(config, vocab_manager, feature_dim)


def create_bert_regression(vocab_manager: VocabManager,
                          hidden_size: int = 768,
                          num_hidden_layers: int = 12,
                          num_attention_heads: int = 12,
                          pooling_method: str = 'mean',
                          **kwargs) -> BertRegression:
    """创建BERT回归模型"""
    config = BertConfig(
        vocab_size=vocab_manager.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        pad_token_id=vocab_manager.pad_token_id,
        **kwargs
    )
    
    return BertRegression(config, vocab_manager, pooling_method)


def create_bert_classification(vocab_manager: VocabManager,
                              num_classes: int,
                              hidden_size: int = 768,
                              num_hidden_layers: int = 12,
                              num_attention_heads: int = 12,
                              pooling_method: str = 'mean',
                              **kwargs) -> BertClassification:
    """创建BERT分类模型"""
    config = BertConfig(
        vocab_size=vocab_manager.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        pad_token_id=vocab_manager.pad_token_id,
        **kwargs
    )
    
    return BertClassification(config, vocab_manager, num_classes, pooling_method) 