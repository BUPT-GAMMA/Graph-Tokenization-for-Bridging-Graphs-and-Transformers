# BERT训练Pipeline - 开发者文档

这是一个支持Token ID序列输入的BERT训练系统，专为需要自定义词表和序列级回归任务而设计。

## 核心特性

- ✅ **Token ID输入**: 直接处理 `[[1,50,8909,...], [...], ...]` 格式的数据
- ✅ **内部词表管理**: 从输入数据自动构建和管理词表
- ✅ **MLM预训练**: 支持Masked Language Modeling预训练
- ✅ **回归任务**: 序列级连续数值预测（平均池化 + MLP）
- ✅ **端到端流程**: 从数据到模型的完整pipeline

## 系统架构

### Pipeline 流程

```
Token ID序列 → 词表构建 → MLM预训练 → 回归微调 → 模型推理
```

### 核心组件

| 组件 | 文件 | 功能 |
|------|------|------|
| 词表管理 | `vocab_manager.py` | 从Token ID统计构建词表，序列编码解码 |
| 模型定义 | `model.py` | BERT MLM模型和回归模型实现 |
| 数据处理 | `data.py` | Token ID序列的数据集和数据加载器 |
| 训练脚本 | `train.py` | MLM预训练和回归微调的训练器 |
| 使用示例 | `example.py` | 完整workflow演示 |

## API接口

### 1. 词表管理 (`vocab_manager.py`)

#### 核心类：`VocabManager`

```python
from vocab_manager import VocabManager, build_vocab_from_sequences

# 从Token ID序列构建词表
token_sequences = [[1,50,8909,234], [42,123,456], ...]
vocab_manager = build_vocab_from_sequences(
    token_sequences, 
    min_freq=2,           # 最小词频
    max_vocab_size=30000  # 最大词表大小
)

# 编码序列
encoded = vocab_manager.encode_sequence([1,50,8909], max_length=128)
# 返回: {'input_ids': tensor, 'attention_mask': tensor}

# 保存/加载词表
vocab_manager.save_vocab("vocab.json")
vocab_manager = VocabManager.load_vocab("vocab.json")
```

#### 主要方法

- `build_vocab(min_freq, max_vocab_size)`: 构建词表
- `encode_sequence(tokens, max_length)`: 编码单个序列  
- `encode_batch(token_sequences)`: 批量编码
- `save_vocab(path)` / `load_vocab(path)`: 保存/加载词表

### 2. 模型定义 (`model.py`)

#### MLM模型：`BertMLM`

```python
from model import create_bert_mlm, BertMLM

# 创建模型
model = create_bert_mlm(
    vocab_manager=vocab_manager,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)

# 前向传播
outputs = model(input_ids, attention_mask, labels)
# 返回: {'logits': tensor, 'loss': tensor}

# 保存/加载
model.save_model("./mlm_model")
model = BertMLM.load_model("./mlm_model")
```

#### 回归模型：`BertRegression`

```python
from model import create_bert_regression, BertRegression

# 创建模型
model = create_bert_regression(
    vocab_manager=vocab_manager,
    hidden_size=768,
    pooling_method='mean'  # 'mean', 'cls', 'max'
)

# 前向传播
outputs = model(input_ids, attention_mask, labels)
# 返回: {'predictions': tensor, 'loss': tensor}

# 预测
predictions = model.predict(input_ids, attention_mask)
```

#### 架构细节

**回归模型架构**（按您的要求）:
```
[B, C, d] → BERT编码 → [B, C, d] → 平均池化 → [B, d] → MLP → [B, 1]
```

**池化策略**:
- `mean`: 平均池化（忽略padding）
- `cls`: 使用[CLS] token
- `max`: 最大池化

### 3. 数据处理 (`data.py`)

#### 数据格式

```json
// MLM数据
{"tokens": [1, 50, 8909, 234]}
{"tokens": [42, 123, 456, 789]}

// 回归数据  
{"tokens": [1, 50, 8909, 234], "label": 2.34}
{"tokens": [42, 123, 456, 789], "label": 1.85}
```

#### 数据加载

```python
from data import load_token_sequences_from_json, create_mlm_dataloader, create_regression_dataloader

# 加载数据
token_sequences = load_token_sequences_from_json("data.jsonl", "tokens")
token_sequences, labels = load_token_sequences_from_json("data.jsonl", "tokens", "label")

# 创建数据加载器
mlm_dataloader = create_mlm_dataloader(
    token_sequences, vocab_manager, 
    batch_size=8, max_length=512, mlm_probability=0.15
)

regression_dataloader = create_regression_dataloader(
    token_sequences, labels, vocab_manager,
    batch_size=8, max_length=512
)
```

#### 核心数据集类

- `MLMDataset`: 自动MLM masking，支持80%[MASK]/10%随机/10%保持不变
- `RegressionDataset`: 序列-标签对，用于回归训练

### 4. 训练脚本 (`train.py`)

#### 高级训练函数

```python
from train import train_mlm, train_regression

# MLM预训练
mlm_model, vocab_manager = train_mlm(
    data_file="mlm_data.jsonl",
    hidden_size=768,
    num_layers=12,
    batch_size=8,
    max_steps=10000,
    save_dir="./mlm_model"
)

# 回归微调
regression_model, vocab_manager = train_regression(
    data_file="regression_data.jsonl", 
    pretrained_mlm_path="./mlm_model/final",  # 可选：加载预训练权重
    hidden_size=768,
    num_layers=12,
    batch_size=8,
    max_steps=1000,
    save_dir="./regression_model"
)
```

#### 训练器类：`Trainer`

```python
from train import Trainer

trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    lr=2e-5,
    max_steps=1000,
    save_dir="./checkpoints"
)
trainer.train()
```

## 命令行使用

### MLM预训练

```bash
python train.py \
    --task mlm \
    --data_file mlm_data.jsonl \
    --hidden_size 768 \
    --num_layers 12 \
    --batch_size 8 \
    --max_steps 10000 \
    --save_dir ./mlm_model
```

### 回归微调

```bash
python train.py \
    --task regression \
    --data_file regression_data.jsonl \
    --pretrained_mlm_path ./mlm_model/final \
    --hidden_size 768 \
    --num_layers 12 \
    --batch_size 8 \
    --max_steps 1000 \
    --save_dir ./regression_model
```

## 完整使用示例

### 1. 基础workflow

```python
# 1. 准备数据
token_sequences = [[1,50,8909,234], [42,123,456,789], ...]
labels = [2.34, 1.85, ...]

# 2. 构建词表
vocab_manager = build_vocab_from_sequences(token_sequences)

# 3. MLM预训练
mlm_model = create_bert_mlm(vocab_manager)
# ... 训练代码

# 4. 回归微调  
regression_model = create_bert_regression(vocab_manager)
# 加载MLM权重: regression_model.bert.load_state_dict(mlm_model.bert.state_dict())
# ... 训练代码

# 5. 推理
predictions = regression_model.predict(input_ids, attention_mask)
```

### 2. 快速开始

```bash
# 运行完整示例
python example.py
```

## 关键设计决策

### 1. 词表管理

- **原始Token ID映射**: 输入的Token ID被映射到内部连续的vocab ID空间
- **频率过滤**: 支持最小词频过滤，低频token被标记为UNK
- **大小限制**: 支持最大词表大小限制，保留高频token
- **特殊Token**: 自动预留PAD(0), UNK(1), CLS(2), SEP(3), MASK(4)

### 2. 模型架构

- **BERT基础**: 基于HuggingFace的BERT实现，确保架构标准性
- **自定义词表**: 动态适应输入数据的词表大小
- **回归头设计**: 严格按照要求实现平均池化+MLP
- **权重初始化**: 使用标准的BERT初始化策略

### 3. 训练流程

- **两阶段训练**: MLM预训练 → 回归微调
- **权重继承**: 回归模型可以加载MLM预训练的BERT权重
- **简化训练器**: 去除复杂配置，专注核心训练逻辑
- **检查点保存**: 支持定期保存和最佳模型保存

## 性能考虑

### 内存优化

- 词表大小限制减少embedding参数
- 支持小batch size减少GPU内存使用
- pin_memory优化数据传输

### 训练效率

- 仅MLM任务，训练速度快于MLM+NSP
- 自动设备检测（CPU/GPU）
- tqdm进度条提供训练反馈

### 推理优化

- `model.eval()`和`torch.no_grad()`减少计算开销
- 批量编码支持高效推理
- 设备自动管理

## 扩展建议

### 功能扩展

1. **多任务学习**: 同时训练多个回归目标
2. **序列标注**: 添加token级别的分类任务
3. **对比学习**: 实现序列相似度计算
4. **知识蒸馏**: 支持大模型到小模型的知识转移

### 性能优化

1. **混合精度**: 使用FP16减少内存和加速训练
2. **梯度累积**: 支持更大的有效batch size
3. **分布式训练**: 多GPU并行训练
4. **模型量化**: 部署优化

### 工程改进

1. **配置管理**: 使用YAML/JSON配置文件
2. **日志系统**: 详细的训练日志和可视化
3. **评估指标**: 添加更多评估指标
4. **数据验证**: 输入数据格式检查和错误处理

## 文件结构

```
minimal_bert/
├── vocab_manager.py     # 词表管理器
├── model.py            # BERT模型定义  
├── data.py             # 数据处理
├── train.py            # 训练脚本
├── example.py          # 使用示例
├── requirements.txt    # 依赖列表
├── README.md          # 本文档
└── backup/            # 原始HF版本备份
    ├── README.md
    └── model.py
```

## 依赖要求

```txt
torch>=1.9.0
transformers>=4.20.0
tqdm>=4.64.0
numpy>=1.21.0
```

## 注意事项

1. **数据格式**: 确保输入的Token ID都是非负整数
2. **内存管理**: 大词表会显著增加内存使用
3. **设备兼容**: 自动检测GPU，但也支持纯CPU训练
4. **模型保存**: 包含模型权重、配置和词表的完整保存
5. **版本兼容**: 建议使用相同版本的PyTorch和Transformers

这个系统专为您的特定需求设计：Token ID输入、内部词表管理、MLM-only预训练、序列级回归任务。所有接口都经过简化，便于理解和扩展。 