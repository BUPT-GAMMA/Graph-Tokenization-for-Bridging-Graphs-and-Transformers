# 图序列化算法架构文档

> **最后更新**：2025-10-30 | **版本**：v2.0  
> **状态**：✅ 已验证与代码完全对齐

## 概述

本模块将图结构转换为可供序列模型使用的token序列。所有序列化器遵循统一接口，并由工厂统一创建与注册。

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| 基类与接口 | `base_serializer.py` | 定义`BaseGraphSerializer`和统一接口 |
| 工厂 | `serializer_factory.py` | 管理序列化器注册和创建 |
| 序列化结果 | `base_serializer.py` | `SerializationResult`类 |

## 已注册的序列化方法

### 图序列化方法（推荐用于图数据）

| 方法名 | 实现文件 | 说明 |
|--------|---------|------|
| `feuler` | `freq_eulerian_serializer.py` | **推荐** 频率引导欧拉回路（原graph_seq） |
| `eulerian` | `eulerian_serializer.py` | 标准欧拉回路序列化 |
| `cpp` | `chinese_postman_serializer.py` | 中国邮路算法 |
| `fcpp` | `freq_chinese_postman_serializer.py` | 频率引导邮路算法 |
| `dfs` | `dfs_serializer.py` | 深度优先搜索遍历 |
| `bfs` | `bfs_serializer.py` | 广度优先搜索遍历 |
| `topo` | `topo_serializer.py` | 拓扑排序（有向图） |

### SMILES序列化方法（用于分子图）

| 方法名 | 实现文件 | 说明 |
|--------|---------|------|
| `smiles` | `smiles_serializer.py` | 默认SMILES字符串 |
| `smiles_1` | `smiles_serializer.py` | SMILES变体1 |
| `smiles_2` | `smiles_serializer.py` | SMILES变体2 |
| `smiles_3` | `smiles_serializer.py` | SMILES变体3 |
| `smiles_4` | `smiles_serializer.py` | SMILES变体4 |

### 图像序列化方法（用于图像/网格图）

| 方法名 | 实现文件 | 说明 |
|--------|---------|------|
| `image_row_major` | `image_row_major_serializer.py` | 行主序遍历 |
| `image_serpentine` | `image_serpentine_serializer.py` | 蛇形遍历 |
| `image_diag_zigzag` | `image_diag_zigzag_serializer.py` | 对角之字形遍历 |

### 查询可用方法

```python
from src.algorithms.serializer.serializer_factory import SerializerFactory

# 获取图序列化方法
graph_methods = SerializerFactory.get_available_serializers()
# → ['smiles', 'smiles_1', 'smiles_2', 'smiles_3', 'smiles_4',
#    'dfs', 'bfs', 'eulerian', 'topo', 'feuler', 'cpp', 'fcpp']

# 获取图像序列化方法  
image_methods = SerializerFactory.get_image_serializers()
# → ['image_row_major', 'image_serpentine', 'image_diag_zigzag']
```

## 统一接口

所有序列化器继承自`BaseGraphSerializer`，提供四个标准方法：

```python
# 1. serialize(graph_data, **kwargs) -> SerializationResult
#    单图单次序列化

# 2. multiple_serialize(graph_data, num_samples, **kwargs) -> SerializationResult
#    单图多次序列化（用于随机性序列化方法）

# 3. batch_serialize(graph_data_list, parallel=True, **kwargs) -> List[SerializationResult]
#    多图批量序列化

# 4. batch_multiple_serialize(graph_data_list, num_samples, parallel=True, **kwargs) -> List[SerializationResult]
#    多图多次序列化
```

## 基本使用示例

### 示例1：直接使用序列化器

```python
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.data.unified_data_factory import get_dataloader
from config import ProjectConfig

# 1. 准备数据
cfg = ProjectConfig()
loader = get_dataloader("qm9test", cfg)
train_data, _ = loader.get_train_data()

# 2. 创建序列化器
serializer = SerializerFactory.create_serializer('feuler')

# 3. 初始化（feuler需要收集全局统计信息）
serializer.initialize_with_dataset(loader, train_data[:100])

# 4. 序列化单个图
result = serializer.serialize(train_data[0])
token_seq, element_seq = result.get_sequence(0)
print(f"Token序列: {token_seq}")
print(f"元素序列: {element_seq}")
```

### 示例2：结合UnifiedDataInterface（推荐）

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

# UDI会自动处理序列化器的初始化和调用
cfg = ProjectConfig()
udi = UnifiedDataInterface(cfg, "qm9test")

# 直接获取序列化结果（UDI内部已完成序列化）
sequences, labels = udi.get_sequences(method="feuler")

# 或获取训练数据
train_seq, val_seq, test_seq, train_y, val_y, test_y = \
    udi.get_training_data(method="feuler", target_property="homo")
```

### 示例3：批量序列化

```python
# 批量序列化（支持并行）
results = serializer.batch_serialize(
    train_data[:100], 
    parallel=True  # 使用多进程加速
)

# 提取所有token序列
all_sequences = [res.get_sequence(0)[0] for res in results]
print(f"平均序列长度: {sum(len(s) for s in all_sequences) / len(all_sequences):.2f}")
```

### 示例4：多次序列化（用于随机性方法）

```python
# 为单个图生成多个不同的序列（如果方法支持随机性）
result = serializer.multiple_serialize(
    train_data[0], 
    num_samples=5
)

# 获取所有生成的序列
for i in range(5):
    token_seq, element_seq = result.get_sequence(i)
    print(f"序列 {i+1}: 长度={len(token_seq)}")
```

## 错误处理

序列化器采用严格的错误处理策略：

```python
# ❌ 图数据缺少必需字段
serializer.serialize({'id': 'test'})  # 缺少 'dgl_graph'
# → 抛出 ValueError: 图数据缺少必需字段 'dgl_graph'

# ❌ 图为空
empty_graph = {'dgl_graph': dgl.graph(([], []))}
serializer.serialize(empty_graph)
# → 抛出 ValueError: 图为空或节点数为0

# ✅ 正确处理
try:
    result = serializer.serialize(graph_data)
except ValueError as e:
    print(f"序列化失败: {e}")
```

## 方法特性与参数

### Feuler（频率引导欧拉 - 推荐）

**参数**：
```python
SerializerFactory.create_serializer('feuler', 
    verbose=False,                  # 是否输出详细日志
    include_edge_tokens=True,       # 是否在序列中包含边token
    omit_most_frequent_edge=True    # 是否省略最高频边类型（减少序列长度）
)
```

**特性**：
- ✅ **确定性**：基于全局频率统计，同一数据集总是产生相同序列
- ✅ **高效**：频率引导确保遍历常见模式
- ✅ **可调**：支持配置边token和频率省略
- ⚠️ **需要初始化**：必须调用`initialize_with_dataset()`收集统计信息

**典型用法**：
```python
# 标准配置（包含边，省略最高频边）
ser = SerializerFactory.create_serializer('feuler')

# 仅节点序列
ser = SerializerFactory.create_serializer('feuler', include_edge_tokens=False)

# 包含所有边token
ser = SerializerFactory.create_serializer('feuler', omit_most_frequent_edge=False)
```

### Eulerian（标准欧拉）

**特性**：
- ✅ **确定性**：通过邻居排序保证跨环境稳定性
- ✅ **简单**：无需统计信息，直接使用

**参数**：与feuler相同，但不使用频率引导

### DFS / BFS

**特性**：
- ✅ **简单快速**：无需初始化
- ⚠️ **序列较长**：可能重复访问节点

### Topo（拓扑排序）

**特性**：
- ✅ **适合有向图**：按依赖关系排序
- ⚠️ **无向图需定向**：会自动为无向边定向

### SMILES系列

**特性**：
- ✅ **分子专用**：利用化学语义
- ⚠️ **仅限分子**：需要RDKit和有效的分子结构

### 图像序列化

**特性**：
- ✅ **适合网格图**：图像、棋盘等规则结构
- ⚠️ **需要网格布局**：图必须具有行列结构

## 返回格式

所有序列化方法返回`SerializationResult`对象：

```python
result = serializer.serialize(graph_data)

# 获取token序列和元素序列
token_seq, element_seq = result.get_sequence(index=0)

# token_seq: List[int] - token ID列表
# element_seq: List[str] - 可读的元素标识列表
#   格式: ['START_NODE_0', 'node_0', 'edge_type_1', 'node_1', ...]
```

**多次序列化结果**：
```python
result = serializer.multiple_serialize(graph_data, num_samples=3)

# 获取第i个变体
for i in range(3):
    token_seq, element_seq = result.get_sequence(i)
```

## 配置集成

序列化方法名在配置文件中指定：

```yaml
# config/default_config.yml
serialization:
  method: feuler  # 默认方法
  multiple_sampling:
    enabled: false
    num_realizations: 1
```

目录结构由方法名确定：
```
processed_data/<dataset>/serialized_data/<method>/single/...
processed_data/<dataset>/serialized_data/<method>/multi_5/...
```

## 设计原则

1. **统一接口**：所有方法实现相同的4个标准方法
2. **确定性优先**：相同输入产生相同输出（除非明确支持随机性）
3. **显式错误**：缺少字段或无效输入立即抛出异常，不使用fallback
4. **职责分离**：序列化器只负责图→序列，不处理BPE或模型训练

## 性能注意事项

### 并行化

```python
# 批量序列化支持多进程（Linux/macOS）
results = serializer.batch_serialize(
    graphs, 
    parallel=True  # 使用fork模式，共享内存
)

# Windows或不支持fork的环境会抛出错误
```

### 内存使用

- **Feuler/Fcpp**：需要预先收集统计信息，内存占用较高
- **DFS/BFS/Eulerian**：无需统计信息，内存占用低
- **批量处理**：建议分批处理大数据集

## 常见问题

### Q: 为什么feuler需要初始化？

Feuler使用全局频率统计来引导遍历，必须先从数据集中收集这些统计信息。

### Q: 如何选择序列化方法？

- **分子图 + 确定性**：`feuler`（推荐）
- **分子图 + 化学语义**：`smiles`
- **一般图 + 简单快速**：`dfs` 或 `bfs`
- **有向图**：`topo`
- **图像/网格**：`image_row_major`

### Q: 序列长度如何优化？

- 设置`omit_most_frequent_edge=True`省略最高频边类型
- 使用BPE压缩（见`src/algorithms/compression/`）

### Q: 如何验证序列化结果？

```python
result = serializer.serialize(graph_data)
token_seq, element_seq = result.get_sequence(0)

print(f"Token序列长度: {len(token_seq)}")
print(f"元素序列: {element_seq[:10]}")  # 查看前10个元素

# 统计信息
stats = result.get_statistics()
print(f"统计信息: {stats}")
```

## 历史命名映射

为保持向后兼容性说明：

| 旧名称 | 新名称 | 说明 |
|-------|--------|------|
| `graph_seq` | `feuler` | 频率引导欧拉回路 |
| `topological` | `topo` | 拓扑排序 |

代码中应统一使用新名称。

## 相关文档

- [`base_serializer.py`](base_serializer.py) - 查看基类完整实现
- [`../compression/README.md`](../compression/README.md) - BPE压缩文档
- [`../../data/README.md`](../../data/README.md) - 数据层文档

---

**文档版本**：v2.0  
**最后更新**：2025-10-30  
**维护者**：发现文档与代码不符请立即修正
