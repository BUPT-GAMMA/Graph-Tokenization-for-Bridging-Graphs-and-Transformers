# TokenizerGraph 项目架构文档

## 概述

TokenizerGraph 是一个基于分子图序列化和BERT预训练的分子性质预测系统。系统支持MLM预训练和下游任务微调两种模式。

## 核心设计原则

### 统一模型创建入口
系统使用单一函数 `create_model_from_udi()` 作为所有模型创建的统一入口，通过参数控制不同的创建行为。

### 二阶段模型创建流程
1. **第1阶段**：创建完整模型（encoder + 任务头），处理权重初始化
2. **第2阶段**：根据需要加载预训练权重，覆盖第1阶段权重

### 配置驱动的权重管理
- 权重重置通过 `config.reset_weights` 统一控制
- 所有权重相关逻辑在encoder创建时处理

## 系统架构

### 主要组件层次

```
训练管道层 (Training Pipelines)
├── src/training/pretrain_pipeline.py    # MLM预训练管道
└── src/training/finetune_pipeline.py    # 微调管道

模型构建层 (Model Building)
├── src/training/model_builder.py        # 模型构建包装器
└── src/models/bert/heads.py             # 统一模型创建核心

核心模型层 (Core Models)
├── src/models/universal_model.py        # 统一模型架构
├── src/models/unified_encoder.py        # 编码器抽象层
├── src/models/unified_task_head.py      # 任务头管理
└── src/models/model_factory.py          # 模型工厂（已弃用）

基础组件层 (Foundation)
├── src/data/unified_data_interface.py   # 数据接口
├── src/training/task_handler.py         # 任务处理器
└── config.py                            # 配置管理
```

## 统一模型创建流程

### 函数接口
```python
def create_model_from_udi(
    udi,                    # UnifiedDataInterface实例
    pretrained_path=None,   # 预训练模型路径
    force_task_type=None    # 强制指定任务类型
):
```

### 参数说明
- `udi`: 统一数据接口，提供数据集信息和配置
- `pretrained_path`: 预训练模型路径，为None时不加载预训练权重
- `force_task_type`: 强制任务类型，用于覆盖UDI推断的任务类型

### 任务类型判断逻辑
```python
if force_task_type is not None:
    task_type = force_task_type          # 使用强制指定类型
else:
    task_type = udi.get_dataset_task_type()  # 从数据集自动推断
```

### 二阶段创建流程

#### 第1阶段：完整模型创建
1. 创建编码器实例（BertEncoder或GTEEncoder）
2. 在编码器初始化时根据 `config.reset_weights` 处理权重重置
3. 根据任务类型确定输出维度和创建任务处理器
4. 创建 UniversalModel 实例，包含编码器和任务头

#### 第2阶段：预训练权重加载
如果提供 `pretrained_path`：
1. 验证预训练模型文件存在性
2. 创建临时编码器实例用于模型加载
3. 加载预训练 UniversalModel
4. 将预训练编码器权重复制到当前模型编码器
5. 覆盖第1阶段的所有编码器参数

## 不同使用模式

### MLM预训练模式
```python
# src/training/pretrain_pipeline.py
model, task_handler = create_model_from_udi(
    udi=udi,
    pretrained_path=None,        # 不加载预训练权重
    force_task_type='mlm'        # 强制MLM任务类型
)
```

特点：
- 任务类型：'mlm'
- 输出维度：词表大小
- 任务头结构：单层Linear投影
- 数据处理：序列级处理，每个token位置预测

### 微调模式
```python
# src/training/finetune_pipeline.py (通过model_builder调用)
model, task_handler = create_model_from_udi(
    udi=udi,
    pretrained_path=auto_resolved,   # 自动解析预训练路径
    force_task_type=None            # 从UDI自动获取任务类型
)
```

特点：
- 任务类型：从数据集推断（'regression', 'classification'等）
- 输出维度：根据任务类型确定
- 任务头结构：多层MLP
- 数据处理：句子级处理，序列池化后预测

## 核心组件详细说明

### UniversalModel
统一模型架构，支持所有任务类型。

**主要属性：**
- `encoder`: 编码器实例（BertEncoder或GTEEncoder）
- `task_head`: 任务头实例（UnifiedTaskHead）
- `task_type`: 任务类型标识
- `pooling_method`: 序列池化方法

**前向传播逻辑：**
- MLM任务：使用 `encoder.get_sequence_output()` 获取序列级表示
- 其他任务：使用 `encoder.encode()` 获取池化后句子级表示

### 编码器层次结构

#### BaseEncoder（抽象基类）
定义编码器统一接口：
- `encode()`: 获取句子级编码表示
- `get_sequence_output()`: 获取序列级编码表示
- `get_hidden_size()`: 获取隐藏层维度

#### BertEncoder（BERT实现）
基于项目内置BERT实现：
- 创建 HuggingFace `BertModel` 实例
- 权重重新初始化在 `__init__` 中根据config处理
- 支持自定义词表大小

#### GTEEncoder（GTE实现）
基于 Alibaba-NLP GTE模型：
- 使用 `AutoModel.from_pretrained()` 加载预训练GTE
- 动态调整词表嵌入层大小
- 权重重新初始化可选择性丢弃预训练权重

### UnifiedTaskHead
任务特定的预测头管理器。

**MLM任务头：**
- 结构：单层Linear投影
- 输入：`[batch_size, seq_len, hidden_size]`
- 输出：`[batch_size, seq_len, vocab_size]`

**其他任务头：**
- 结构：可配置多层MLP
- 输入：`[batch_size, hidden_size]`
- 输出：`[batch_size, output_dim]`
- 配置参数：hidden_ratio, activation, dropout

## 权重管理机制

### 权重初始化类型
1. **默认初始化**：使用模型原生初始化参数
2. **重新初始化**：通过 `config.reset_weights=True` 启用，丢弃预训练知识
3. **预训练权重**：通过第2阶段加载，覆盖前两种初始化

### 权重处理时机
- **编码器创建时**：处理reset权重（第1阶段）
- **模型加载时**：复制预训练权重（第2阶段）

### 权重覆盖优先级
预训练权重 > 重新初始化权重 > 默认初始化权重

## 设计决策和原则

### 任务输出维度硬编码
**设计决策**：任务输出维度在 `create_model_from_udi()` 中硬编码，不使用动态推断。

**合理性**：
- 标准任务类型的输出维度是确定的（回归=1，二分类=2，多分类=数据集类别数）
- 硬编码避免了运行时的复杂推断逻辑和潜在错误
- 提高了代码的可读性和可维护性
- 如果数据集与标准任务不符，应在数据准备阶段解决，而非在模型层动态适配

### Fail-Fast错误处理原则
**设计决策**：系统采用严格的fail-fast错误处理，不使用静默错误恢复。

**合理性**：
- 科研代码需要确保实验结果的可靠性，静默错误会掩盖问题
- 明确的异常信息有助于快速定位和解决问题
- 避免了错误传播导致的不可预测行为
- 符合"宁可停止运行也不产生错误结果"的科研软件原则

### 配置项统一化
**设计决策**：统一使用 `reset_weights` 配置项。

**变更理由**：
- 避免配置项命名不一致造成的用户混淆
- `reset_weights` 语义更清晰，表示"重置权重到初始状态"
- 简化配置管理逻辑，减少fallback处理

## 配置管理

### 关键配置项
- `config.encoder.type`: 编码器类型（'bert', 'Alibaba-NLP/gte-multilingual-base'）
- `config.reset_weights`: 权重重置开关
- `config.bert.architecture.*`: BERT模型结构参数
- `config.serialization.method`: 序列化方法

### 配置访问模式
所有配置通过统一的 `ProjectConfig` 实例访问，避免硬编码配置值。

## 数据流向

### 预训练数据流
```
输入: token_ids [batch_size, seq_len]
  ↓
编码器: sequence_output [batch_size, seq_len, hidden_size]
  ↓
MLM任务头: logits [batch_size, seq_len, vocab_size]
```

### 微调数据流
```
输入: token_ids [batch_size, seq_len]
  ↓
编码器: sequence_output [batch_size, seq_len, hidden_size]
  ↓
池化操作: pooled [batch_size, hidden_size]
  ↓
任务头: predictions [batch_size, output_dim]
```

## 文件结构说明

### 核心创建逻辑
- `src/models/bert/heads.py`: 统一模型创建入口实现

### 训练管道
- `src/training/pretrain_pipeline.py`: MLM预训练完整流程
- `src/training/finetune_pipeline.py`: 下游任务微调流程
- `src/training/model_builder.py`: 模型构建包装和路径解析

### 模型组件
- `src/models/universal_model.py`: 统一模型架构实现
- `src/models/unified_encoder.py`: 编码器抽象和具体实现
- `src/models/unified_task_head.py`: 任务头管理器
- `src/models/model_factory.py`: 历史模型工厂（已弃用）

### 支持组件
- `src/data/unified_data_interface.py`: 数据访问统一接口
- `src/training/task_handler.py`: 任务特定损失函数和指标
- `config.py`: 项目配置管理

## 调用关系图

### 预训练调用链
```
train_bert_mlm()
  ↓
create_model_from_udi(force_task_type='mlm', pretrained_path=None)
  ↓
_create_complete_model() → create_encoder() → BertEncoder/GTEEncoder
  ↓
UniversalModel(task_type='mlm') → UnifiedTaskHead(MLM头)
```

### 微调调用链
```
run_finetune()
  ↓
build_task_model()
  ↓
_resolve_pretrained_path_internal() + create_model_from_udi()
  ↓
_create_complete_model() → create_encoder() → BertEncoder/GTEEncoder
  ↓
UniversalModel(task_type=auto) → UnifiedTaskHead(任务头)
  ↓
_load_and_copy_pretrained_weights() (可选)
```

## 扩展点

### 新增编码器类型
在 `src/models/unified_encoder.py` 中：
1. 继承 `BaseEncoder` 创建新编码器类
2. 在 `create_encoder()` 工厂函数中添加分发逻辑

### 新增任务类型
在 `src/models/bert/heads.py` 中：
1. 在任务类型判断逻辑中添加新类型处理
2. 在 `src/models/unified_task_head.py` 中实现任务头结构
3. 在 `src/training/task_handler.py` 中实现损失函数

### 新增权重初始化策略
在编码器的 `__init__` 方法中扩展权重初始化逻辑。

---

*本文档描述的是当前系统的实际架构状态，不涉及历史演进或性能评价。*