# GTE模型简单集成方案

## 🎯 核心理念

**用户需求分析**：
- encoder就是一个model：输入sequence，输出vector
- 需要一个抽象，让BERT和GTE都是抽象的实现
- 利用现有架构，不瞎规划主流程修改
- 支持项目的14个数据集，不局限于分子图

## 🏗️ 现有架构分析

### 当前工作流程
```
pretrained_model (backbone) 
    ↓ 
create_model_from_udi()
    ↓
BertUnified + TaskHandler
```

### 关键发现
1. **`pretrained_model`是backbone**，提供encoding能力
2. **`create_model_from_udi`**将backbone包装成任务模型
3. **权重复制机制**：`model.bert.load_state_dict(pretrained_model.bert.state_dict())`
4. **TaskHandler**处理任务相关逻辑（损失函数、指标等）

## 📋 集成方案设计

### 1. 创建抽象Encoder接口

```python
# src/models/base_encoder.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseEncoder(nn.Module, ABC):
    """抽象encoder接口：输入sequence，输出vector"""
    
    @abstractmethod
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """编码方法：sequence → vector"""
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """编码维度"""
        pass
    
    @property
    @abstractmethod
    def vocab_manager(self):
        """词汇表管理器"""
        pass
    
    @property 
    @abstractmethod
    def config(self):
        """模型配置"""
        pass
```

### 2. BERT Encoder实现

```python
# src/models/bert/bert_encoder.py
from ..base_encoder import BaseEncoder

class BertEncoder(BaseEncoder):
    """BERT实现的encoder"""
    
    def __init__(self, bert_mlm_model):
        super().__init__()
        self.bert_model = bert_mlm_model
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """使用BERT编码：sequence → vector"""
        outputs = self.bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 平均池化
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        masked_output = sequence_output * attention_mask.unsqueeze(-1)
        pooled = masked_output.sum(dim=1) / lengths
        
        return pooled
    
    @property
    def hidden_size(self) -> int:
        return self.bert_model.config.hidden_size
    
    @property 
    def vocab_manager(self):
        return self.bert_model.vocab_manager
    
    @property
    def config(self):
        return self.bert_model.config
```

### 3. GTE Encoder实现

```python
# src/models/gte/gte_encoder.py
import torch
from transformers import AutoModel
from ..base_encoder import BaseEncoder

class GTEEncoder(BaseEncoder):
    """GTE实现的encoder"""
    
    def __init__(self, vocab_manager, config_dict=None):
        super().__init__()
        self._vocab_manager = vocab_manager
        
        # 加载预训练GTE模型 
        self.gte_model = AutoModel.from_pretrained(
            'Alibaba-NLP/gte-multilingual-base',
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True,
            torch_dtype=torch.float16
        )
        
        # 简单配置对象
        self._config = type('GTEConfig', (), {
            'hidden_size': 768,
            'vocab_size': vocab_manager.vocab_size,
            'max_position_embeddings': 8192
        })()
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """使用GTE编码：sequence → vector"""
        # GTE直接返回sentence embedding
        return self.gte_model(input_ids=input_ids, attention_mask=attention_mask)
    
    @property
    def hidden_size(self) -> int:
        return 768  # GTE固定768维
    
    @property
    def vocab_manager(self):
        return self._vocab_manager
    
    @property  
    def config(self):
        return self._config
```

### 4. 统一Encoder工厂

```python
# src/models/encoder_factory.py
def create_encoder(encoder_type: str, pretrained_model=None, vocab_manager=None):
    """统一encoder创建接口"""
    
    if encoder_type == 'bert':
        from .bert.bert_encoder import BertEncoder
        if pretrained_model is None:
            raise ValueError("BERT encoder需要pretrained_model")
        return BertEncoder(pretrained_model)
        
    elif encoder_type == 'gte':
        from .gte.gte_encoder import GTEEncoder
        if vocab_manager is None:
            raise ValueError("GTE encoder需要vocab_manager")
        return GTEEncoder(vocab_manager)
        
    else:
        raise ValueError(f"不支持的encoder类型: {encoder_type}")
```

### 5. 修改create_model_from_udi

```python
# 在src/models/bert/heads.py中修改create_model_from_udi
def create_model_from_udi(udi, encoder, pooling_method: str = 'mean'):
    """
    根据UDI创建模型（支持抽象encoder）
    
    Args:
        udi: UnifiedDataInterface实例
        encoder: BaseEncoder实例（BERT或GTE）
        pooling_method: 池化方法
    """
    # 从UDI获取任务信息
    task_handler = create_task_handler(udi)
    
    # 创建统一模型
    model = create_unified_model(
        vocab_manager=encoder.vocab_manager,
        hidden_size=encoder.hidden_size,
        num_hidden_layers=getattr(encoder.config, 'num_hidden_layers', 4),
        num_attention_heads=getattr(encoder.config, 'num_attention_heads', 8),  
        intermediate_size=getattr(encoder.config, 'intermediate_size', 2048),
        pooling_method=pooling_method,
        dropout=getattr(encoder.config, 'dropout', 0.1),
        max_position_embeddings=getattr(encoder.config, 'max_position_embeddings', 512),
        layer_norm_eps=getattr(encoder.config, 'layer_norm_eps', 1e-12),
        output_dim=task_handler.output_dim,
    )
    
    # 复制encoder权重到统一模型
    if hasattr(encoder, 'bert_model') and hasattr(encoder.bert_model, 'bert'):
        # BERT情况
        model.bert.load_state_dict(encoder.bert_model.bert.state_dict())
    elif hasattr(encoder, 'gte_model'):
        # GTE情况：需要权重映射
        model.bert = encoder.gte_model  # 直接替换backbone
    
    return model, task_handler
```

### 6. 修改训练管道的backbone加载

```python
# 在src/training/finetune_pipeline.py中修改
def load_pretrained_backbone(config: ProjectConfig, pretrained_dir: Optional[str] = None):
    """统一的预训练encoder加载"""
    
    encoder_type = getattr(config, 'encoder_type', 'bert')
    
    if encoder_type == 'bert':
        # 现有BERT逻辑不变
        bert_model = _load_bert_backbone(config, pretrained_dir)
        from src.models.encoder_factory import create_encoder
        return create_encoder('bert', bert_model)
        
    elif encoder_type == 'gte':
        # 新增GTE逻辑  
        udi = UnifiedDataInterface(config=config, dataset=config.dataset.name)
        vocab_manager = udi.get_vocab(method=config.serialization.method)
        from src.models.encoder_factory import create_encoder
        return create_encoder('gte', vocab_manager=vocab_manager)
        
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")
```

## 🔧 配置支持

```yaml
# config/default_config.yml 只需简单添加
encoder_type: bert  # bert | gte

# GTE相关配置（可选）
gte:
  optimization:
    unpad_inputs: true
    use_memory_efficient_attention: true
    torch_dtype: "float16"
```

## 📋 实施步骤

### 第1步：创建抽象接口
- [ ] 实现`BaseEncoder`抽象类
- [ ] 实现`BertEncoder`包装现有BERT
- [ ] 实现`GTEEncoder`加载GTE模型

### 第2步：集成现有系统
- [ ] 创建`encoder_factory.py`
- [ ] 修改`create_model_from_udi`支持抽象encoder
- [ ] 修改`load_pretrained_backbone`使用encoder工厂

### 第3步：测试验证
- [ ] 在14个数据集上测试BERT encoder（确保向后兼容）
- [ ] 在qm9test上测试GTE encoder
- [ ] 性能对比测试

## 🎯 核心优势

### 简洁性
- **不修改主流程**：利用现有的`create_model_from_udi`机制
- **encoder就是encoder**：输入sequence，输出vector，仅此而已
- **抽象简单清晰**：只定义必要的接口方法

### 兼容性  
- **向后兼容**：现有BERT模型完全不变
- **支持14个数据集**：不局限于分子图
- **配置统一**：通过`encoder_type`简单切换

### 可扩展性
- **易于添加新encoder**：只需实现`BaseEncoder`接口
- **config隐藏实现细节**：模型特定逻辑在encoder内部处理

---

**核心理念**：Keep It Simple. encoder就是输入sequence输出vector的黑盒，抽象接口隐藏实现差异，利用现有架构不瞎改主流程。
