# TokenizerGraph 项目高层设计总结

## 项目核心理念

TokenizerGraph是一个通用的图tokenization框架，通过将任意图结构转换为Transformer可以处理的序列表示。该框架结合了可逆序列化算法和BPE压缩技术，为图数据的深度学习应用提供了一个系统性的解决方案。

### 框架核心组件

1. **可逆序列化算法**
   - 基于欧拉路径等信息论可逆方法
   - 确保图的拓扑信息完全保留
   - 支持确定性和概率性序列化策略

2. **结构感知tokenization机制**啊？
   - 使用数据集级别的结构统计信息引导序列化
   - 通过BPE学习有意义的子结构模式
   - 生成稳定且信息丰富的离散表示

3. **标准Transformer兼容性**
   - 产生的token序列可直接输入标准Transformer模型
   - 支持现有的预训练语言模型生态系统
   - 无需定制图神经网络架构

## 核心数据流程

### 当前实现：Offline预处理模式

```
原始图数据 → 图结构 → 序列化 → BPE压缩 → 持久化存储
                                      ↓
                               训练数据 ← 加载处理后的数据
```

**优势：**
- 预处理一次，训练多次使用
- 序列化算法可以任意复杂度，无性能压力
- 支持多重采样（一个图生成多个序列变体）

**局限性：**
- 数据预处理时间较长
- 存储空间占用较大
- 难以支持在线数据增强

### 设计目标：Online处理模式

```
原始图数据 → 图结构 → 实时序列化 → BPE压缩 → 模型输入
```

**预期优势：**
- 零预处理开销，支持实时数据加载
- 灵活的数据增强策略
- 更适合大规模数据集和在线学习场景

**当前挑战：**
- 序列化算法纯Python实现，性能不足以支持online模式
- BPE算法已优化为C++后端，可以支持online处理
- 计划通过将图结构转换为与库无关的格式来优化序列化性能


注意上述说的问题。我们在论文中讲我们的实践的时候，应该以online的模式来讲。因为没有采用online只是因为性能问题，而这个性能的问题其实是可以解决的。并且我们下一步就会去解决。

## 核心算法组件

### 1. 序列化算法族

#### 频率引导欧拉回路算法 (Feuler)

**核心思想：**
```python
def frequency_guided_serialization(graph):
    # 1. 预先收集数据集级别的三元组频率
    global_stats = collect_triplet_frequencies(dataset)

    # 2. 序列化时使用频率引导遍历
    def choose_next_edge(current_node, candidates):
        return max(candidates,
                  key=lambda edge: get_triplet_frequency(
                      get_node_type(current_node),
                      get_edge_type(edge),
                      get_node_type(get_neighbor(edge))
                  ))

    # 3. 生成确定性序列
    return eulerian_path_with_frequency_guidance(graph, choose_next_edge)
```

**优势：**
- 相同图始终产生相同序列
- 频率信息提供结构引导
- 支持并行统计收集

#### 标准欧拉回路算法 (Eulerian)

**核心思想：**
```python
def standard_eulerian_serialization(graph):
    # 使用经典Hierholzer算法
    # 随机选择起始点和遍历顺序
    return hierholzer_algorithm(graph, random_start=True)
```

**特点：**
- 实现简单，计算效率高
- 结果有一定随机性
- 适合快速原型验证


对于cpp（中国邮递员）算法也是类似的，他的经典实现是一个很复杂的算法。然后我们也为他加入了可以使用频率引导的方式



其次，对于bsdfs拓扑排序，smile之类的方法。我们也进行了实现。


所以说在用伪代码或者说在文章中从比较高层次的来讲，这相关的实现的时候，应该是不涉及算法具体细节的，或者说是从一些不同的序列化方法都存在的环节或者步骤来讲的。只有到相应的章节需要介绍每种序列化方法的时候，可能才会涉及其细节。

### 2. BPE压缩算法

**核心优化：**
```python
class OptimizedBPE:
    def __init__(self):
        self.pair_freqs = defaultdict(int)  # 增量频率表

    def train(self, sequences):
        # 1. 一次性构建初始频率表
        self._build_initial_frequencies(sequences)

        # 2. 增量合并和频率更新
        for _ in range(self.num_merges):
            best_pair = self._find_most_frequent_pair()
            self._merge_and_update_frequencies(sequences, best_pair)

    def _merge_and_update_frequencies(self, sequences, pair):
        # 单次遍历完成合并和频率更新
        # 原地修改序列，增量更新频率表
        pass
```


对于b于b算法，虽然我们做了一些改进，比如说把训练时的合并和频率更新一起做，以及用cpp来实现。有了相当的速度提升，但是这相关的部分可能不是我们这个论文主要要讲的。所以这一部分其实没有必要说的太多。可能就是叙述一下经典的bpe就可以了。不用去特别讲我们这个bpe有什么区别

## 模型架构

### BERT-Small配置

```python
# 为分子序列化数据优化的配置
bert_config = {
    'hidden_size': 512,
    'num_attention_heads': 8,
    'num_hidden_layers': 4,
    'intermediate_size': 2048,
    'max_position_embeddings': 64,  # 分子序列长度
    'vocab_size': 动态确定  # 基于BPE词汇表
}
```

### GTE架构 (Graph Transformer Enhanced)

- 基于近期Transformer改进的新架构
- 参数量与BERT-base相当
- 针对分子序列化数据进行专门优化

## 完整流程伪代码

### 数据预处理阶段 (Offline)

```python
def prepare_data_pipeline(dataset_name, serialization_method):
    # 1. 初始化组件
    config = ProjectConfig()
    dataloader = get_dataloader(dataset_name, config)
    serializer = SerializerFactory.create_serializer(serialization_method)
    bpe_compressor = StandardBPECompressor()

    # 2. 加载原始数据
    graphs, splits = dataloader.get_all_data_with_indices()

    # 3. 序列化处理
    if serialization_method.startswith('f'):  # 频率引导算法
        serializer.initialize_with_dataset(dataloader, graphs)

    # 4. 批量序列化
    sequences = serializer.batch_serialize(graphs)

    # 5. BPE训练和压缩
    bpe_compressor.train(sequences)
    compressed_sequences = [bpe_compressor.encode(seq) for seq in sequences]

    # 6. 持久化存储
    save_processed_data(compressed_sequences, splits, bpe_compressor)
```


上面这个伪代码可以说是相当的从coding角度来讲的，那么我们对于论文中的伪代码显然不能这么写。更应该是结合我们在preliminary里面定义的。那些数学一点的符号，从一些比较数学的角度来写我们的这个伪代码。

### 训练阶段

```python
def training_pipeline(dataset_name, serialization_method, model_type):
    # 1. 加载预处理数据
    train_data, val_data, test_data = load_processed_data(
        dataset_name, serialization_method
    )

    # 2. 初始化模型
    if model_type == 'bert':
        model = BERTModel(config.bert)
    elif model_type == 'gte':
        model = GTEModel(config.gte)

    # 3. 预训练阶段 (MLM任务)
    pretrain_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        mlm_probability=0.15,
        epochs=config.training.epochs
    )

    # 4. 微调阶段 (下游任务)
    finetune_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        task_type=get_task_type(dataset_name),
        target_property=config.task.target_property
    )

    # 5. 评估
    evaluate_model(model, test_data)
```

### 推理阶段

```python
def inference_pipeline(graph, trained_model, serialization_method):
    # 1. 序列化图
    serializer = get_trained_serializer(serialization_method)
    sequence = serializer.serialize(graph)

    # 2. BPE压缩
    bpe_encoder = get_trained_bpe_encoder(serialization_method)
    compressed_sequence = bpe_encoder.encode(sequence)

    # 3. 模型推理
    model_input = prepare_model_input(compressed_sequence)
    prediction = trained_model.predict(model_input)

    return prediction
```

## 框架实现特性

### 1. 架构设计
- **数据层：** 统一的DataLoader接口，支持新数据集快速接入
- **算法层：** 插件化的序列化算法，支持算法的灵活组合
- **模型层：** 统一的Encoder接口，支持多种Transformer架构

### 2. 性能特性
- **并行处理：** 多进程序列化，充分利用多核CPU
- **增量算法：** BPE的增量频率更新，提升训练效率
- **内存优化：** 稀疏统计存储和原地修改算法

### 3. 扩展机制
- **新数据集：** 实现BaseDataLoader接口
- **新算法：** 继承BaseGraphSerializer
- **新任务：** 在UDI中添加任务特定的配置和损失函数

### 4. 实现特性
- **确定性：** 相同输入始终产生相同输出
- **可重现性：** 单一配置源确保实验可重现
- **可维护性：** 清晰的代码结构和统一的接口设计

## 未来优化方向

### 1. Online模式实现
```python
# 当前：Offline预处理
sequences = preprocess_and_cache_all(graphs)

# 目标：Online处理
def __getitem__(self, idx):
    graph = self.graphs[idx]
    sequence = self.serializer.serialize(graph)  # 实时序列化
    compressed = self.bpe_encoder.encode(sequence)  # 实时压缩
    return compressed
```

### 2. 序列化算法优化
- 将核心序列化逻辑迁移到C++实现
- 设计与图库无关的数据格式
- 支持GPU加速的序列化算法

### 3. 扩展应用场景
- 支持更多图类型和领域应用
- 集成多种结构表示方法
- 支持多任务联合学习

这个高层设计总结提供了项目核心思想的完整概览，为理解和扩展TokenizerGraph项目提供了重要指导。项目的设计哲学强调确定性、性能和可扩展性，为图数据深度学习应用提供了系统性的解决方案。
