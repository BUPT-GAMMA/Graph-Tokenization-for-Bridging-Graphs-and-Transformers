# TokenizerGraph 项目详细技术文档

## 项目概述

TokenizerGraph是一个基于图序列化的分子属性预测框架，通过将分子图转换为序列数据，并使用Transformer架构进行特征学习和属性预测。项目采用模块化设计，支持多种序列化算法、BPE压缩以及不同的模型架构。

## 核心架构设计

### 1. 数据层 (Data Layer)

#### 1.1 BaseDataLoader 基类

位置：`src/data/base_loader.py`

**核心功能：**
- 提供统一的抽象接口，定义所有数据集加载器必须实现的方法
- 管理数据集的基本信息（训练/验证/测试划分、标签提取等）
- 提供token映射和管理功能

**关键抽象方法：**
```python
# 数据加载
_load_processed_data() -> Tuple[List, List, List]  # 加载训练/验证/测试数据
_extract_labels(data: List) -> List                # 从数据中提取标签

# Token管理接口
get_node_attribute(graph, node_id) -> int          # 获取节点关键属性
get_edge_attribute(graph, edge_id) -> int          # 获取边关键属性
get_node_token(graph, node_id) -> List[int]        # 获取节点token序列
get_edge_token(graph, edge_id) -> List[int]        # 获取边token序列

# 批量操作接口
get_node_tokens_bulk(graph, node_ids) -> List[List[int]]
get_edge_tokens_bulk(graph, edge_ids) -> List[List[int]]
get_graph_node_token_ids(graph) -> torch.Tensor
get_graph_edge_token_ids(graph) -> torch.Tensor
```

#### 1.2 UnifiedDataInterface (UDI)

位置：`src/data/unified_data_interface.py`

**核心功能：**
- 作为数据层的统一包装接口
- 管理序列化结果的缓存和读取
- 协调DataLoader和序列化算法之间的交互
- 提供BPE编码器的管理和接口

**主要方法：**
```python
# 核心数据读取接口
get_sequences(method: str) -> Tuple[List[Tuple[int, List[int]]], List[Dict]]
get_sequences_by_splits(method: str) -> Tuple[6 elements]

# 序列化管理
prepare_serialization(method: str) -> Path
_build_and_persist_serialization(method: str) -> Path

# BPE管理
get_bpe_codebook(method: str) -> Dict
get_bpe_encoder(method: str) -> BPEEngine
save_bpe_codebook(method: str, merge_rules, vocab_size) -> Path
```

### 2. 算法层 (Algorithm Layer)

#### 2.1 序列化算法 (Serialization)

位置：`src/algorithms/serializer/`

**基类：BaseGraphSerializer**

位置：`src/algorithms/serializer/base_serializer.py`

**核心特性：**
- 统一的序列化接口和流程
- 多进程并行处理支持（使用fork上下文）
- 统计信息收集和频率引导
- 连通分量处理

**主要序列化器实现：**

1. **FeulerSerializer** (`freq_eulerian_serializer.py`)
   - 基于频率引导的欧拉回路算法
   - 在序列化前收集数据集级别的三元组频率统计
   - 使用频率信息引导遍历顺序，确保序列化结果的确定性

2. **EulerianSerializer** (`eulerian_serializer.py`)
   - 标准欧拉回路算法
   - 不使用频率引导，随机性更强

3. **ChinesePostmanSerializer** (`chinese_postman_serializer.py`)
   - 中国邮递员算法
   - 确保所有边都被遍历至少一次

4. **其他序列化器：**
   - BFS/DFS序列化器
   - 图像相关序列化器（行主序、蛇形等）
   - SMILES序列化器

**序列化流程：**
```python
# 1. 初始化序列化器（收集统计信息）
serializer.initialize_with_dataset(dataloader, graph_data_list)

# 2. 序列化单个图
result = serializer.serialize(graph_data, **kwargs)
# 返回：SerializationResult(token_sequences, element_sequences, id_mapping)

# 3. 批量序列化
results = serializer.batch_serialize(graph_data_list, parallel=True)

# 4. 多重采样序列化
results = serializer.batch_multiple_serialize(graph_data_list, num_samples=5)
```

#### 2.2 BPE压缩算法 (BPE Compression)

位置：`src/algorithms/compression/`

**核心实现：BPEEngine**

位置：`src/algorithms/compression/bpe_engine.py`

**架构特性：**
- 统一的高性能BPE训练+编码引擎
- 支持多种后端：C++（推荐）、Numba加速、Python兼容
- 灵活的编码策略：支持topk、random、gaussian等采样模式
- 与minBPE算法语义等价的实现

**后端选择：**
- **C++后端** (`cpp_bpe_backend.py`): 高性能生产环境推荐
- **Numba后端** (`numba_bpe_train.py`): CPU加速选项
- **Python后端** (`main_bpe.py`): 兼容性备用选项

**训练和编码流程：**
```python
# 1. 初始化BPE引擎（选择后端）
engine = BPEEngine(
    train_backend="cpp",      # 训练后端：cpp/numba/python
    encode_backend="cpp",     # 编码后端：cpp
    encode_rank_mode="all"    # 编码模式：all/topk/random/gaussian
)

# 2. 训练（输入必须是List[List[int]]）
stats = engine.train(
    token_sequences,
    num_merges=2000,
    min_frequency=2
)

# 3. 构建编码器
engine.build_encoder()

# 4. 编码
encoded = engine.encode(token_sequence)
batch_encoded = engine.batch_encode(token_sequences)
```

### 3. 模型层 (Model Layer)

位置：`src/models/`

#### 3.1 BERT模型架构

位置：`src/models/bert/`

**配置参数：**
```python
# BERT-Small 配置
d_model = 512          # 隐藏层维度
n_heads = 8           # 注意力头数
n_layers = 4          # Transformer层数
d_ff = 2048          # 前馈网络维度
max_seq_length = 64   # 最大序列长度
```

**核心组件：**
- **配置管理：** `config.py` - BERT模型配置
- **词汇管理：** `vocab_manager.py` - 词汇表管理和特殊token处理
- **数据处理：** `data.py` - 数据批处理和掩码语言模型任务
- **变换器：** `transforms.py` - 数据预处理变换

#### 3.2 GTE模型架构

位置：`src/models/gte/`

- 基于近期Transformer改进的架构
- 参数量与BERT-base相当
- 针对分子序列化数据优化

#### 3.3 统一编码器接口

位置：`src/models/unified_encoder.py`

- 提供统一的编码器接口
- 支持BERT和GTE两种架构的切换
- 封装预训练和微调逻辑

### 4. 训练层 (Training Layer)

位置：`src/training/`

#### 4.1 预训练流水线

**位置：** `src/training/pretrain_pipeline.py`

**核心功能：**
- MLM (Masked Language Model) 任务实现
- 支持分布式训练（多GPU）
- 集成WandB和ClearML日志记录
- 自动学习率调度和early stopping

**预训练流程：**
```python
# 1. 数据准备
train_sequences, val_sequences, test_sequences = udi.get_training_data_flat(method)

# 2. 模型初始化
model = BERTModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 3. MLM训练
for epoch in range(num_epochs):
    # 前向传播和损失计算
    outputs = model(input_ids, attention_mask)
    loss = mlm_loss(outputs.logits, labels, mask_positions)

    # 反向传播
    loss.backward()
    optimizer.step()
```

#### 4.2 微调流水线

**核心功能：**
- 支持回归和分类任务
- 自动任务类型检测
- 多目标任务支持
- 类别权重平衡

### 5. 配置管理系统

位置：`config.py`

#### 5.1 ProjectConfig 类

**核心特性：**
- 单一配置源原则
- 支持YAML配置文件加载
- 命令行参数覆盖
- 自动路径解析和目录创建
- 配置验证和类型转换

**配置层次结构：**
```yaml
# 系统配置
system:
  device: auto
  seed: 42

# 数据集配置
dataset:
  name: qm9
  limit: null

# 序列化配置
serialization:
  method: feuler
  bpe:
    num_merges: 2000
    min_frequency: 100

# BERT模型配置
bert:
  architecture:
    hidden_size: 512
    num_attention_heads: 8
    num_hidden_layers: 4

# 训练配置
training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 100
```

## 数据流程

### 1. 原始数据 -> 图数据

```
Raw Dataset Files
    ↓ [DataLoader]
DGL Graphs + Properties
    ↓ [UDI]
Unified Data Interface
```

### 2. 图数据 -> 序列化

```
DGL Graphs
    ↓ [Serializer.initialize_with_dataset()]
Collect Statistics (optional, only for frequency-guided methods)
    ↓ [Serializer.serialize()]
Token Sequences + Element Mappings
    ↓ [UDI Cache]
Persistent Storage
```

### 3. 序列化 -> BPE压缩

```
Token Sequences
    ↓ [BPE Compressor.train()]
Learn Merge Rules
    ↓ [BPE Compressor.encode()]
Compressed Sequences
    ↓ [Vocab Manager]
BERT-compatible Vocab
```

### 4. 压缩序列 -> 模型训练

```
Compressed Sequences
    ↓ [Data Collator]
Batched Tensors (input_ids, attention_mask, labels)
    ↓ [BERT/GTE Encoder]
Sequence Representations
    ↓ [MLM Head / Task Head]
Predictions
```

## 关键实现细节

### 1. 多进程并行处理

**位置：** `src/algorithms/serializer/base_serializer.py`

**特性：**
- 使用Linux fork上下文确保内存安全
- 支持序列化统计收集的并行化
- 自动CPU核心数检测和负载均衡
- 严格保序输出

### 2. 频率引导序列化

**位置：** `src/algorithms/serializer/freq_eulerian_serializer.py`

**算法流程：**
```python
# 1. 全局统计收集
for graph in dataset:
    extract_triplet_frequencies(graph)
    # (node_type_src, edge_type, node_type_dst) -> frequency

# 2. 序列化时使用频率引导
def find_next_edge(current_node, available_edges):
    # 选择频率最高的边类型进行遍历
    return max(available_edges, key=lambda e: get_edge_frequency(e))
```

### 3. BPE优化实现

**位置：** `src/algorithms/compression/main_bpe.py`

**优化策略：**
```python
# 1. 增量频率更新
self.pair_freqs = defaultdict(int)

# 2. 集成合并和更新
def _optimized_merge_and_update(self, id_sequences, pair, new_id):
    # 单次遍历完成合并和频率更新
    for seq in id_sequences:
        # 原地修改序列
        # 增量更新频率表
```

### 4. 统一损失函数管理

**位置：** `src/data/unified_data_interface.py`

**支持的任务类型：**
- MLM (Masked Language Model)
- 回归 (Regression)
- 分类 (Classification)
- 多标签分类 (Multi-label Classification)
- 多目标回归 (Multi-target Regression)

## 使用指南

### 1. 数据预处理

```bash
# 运行完整数据预处理
python prepare_data_new.py \
  --dataset qm9 \
  --method feuler \
  --bpe_num_merges 2000 \
  --bpe_min_frequency 100 \
  --multiple_samples 1
```

### 2. 预训练

```bash
# BERT预训练
python run_pretrain.py \
  --dataset qm9 \
  --method feuler \
  --experiment_group my_experiment \
  --experiment_name bert_pretrain \
  --bpe_encode_rank_mode all \
  --epochs 200 \
  --batch_size 256 \
  --learning_rate 1e-4
```

### 3. 微调

```bash
# 下游任务微调
python run_finetune.py \
  --dataset qm9 \
  --method feuler \
  --task regression \
  --target_property homo \
  --model_path /path/to/pretrained/model \
  --batch_size 64 \
  --learning_rate 2e-5
```

## 性能和扩展性

### 1. 性能优化

- **序列化：** 多进程并行处理，支持大规模数据集
- **BPE：** 增量更新算法，训练时间减少75%
- **训练：** 支持多GPU分布式训练
- **内存：** 稀疏统计存储，减少内存占用

### 2. 扩展性设计

- **新数据集：** 继承BaseDataLoader实现自定义加载器
- **新序列化算法：** 继承BaseGraphSerializer实现自定义算法
- **新模型架构：** 通过UnifiedEncoder接口集成
- **新任务类型：** 在UDI中添加任务特定的损失函数

## 总结

TokenizerGraph项目采用模块化设计，通过清晰的层次分离实现了从分子图到属性预测的完整流水线。项目强调：

1. **统一接口：** 所有组件都遵循统一的抽象接口
2. **性能优化：** 多进程并行和增量算法确保高效处理
3. **可扩展性：** 模块化设计支持轻松添加新功能
4. **可重现性：** 单一配置源和确定性算法确保实验可重现

这种设计使得项目既适合快速原型开发，也能支持大规模生产部署。
