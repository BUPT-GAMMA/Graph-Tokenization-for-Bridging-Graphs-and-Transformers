# 数据层架构文档

> **重要提示**：本文档已更新至最新版本（2025-10），准确反映当前代码实现。
> 详细的设计理念和演进历史请参考 [`../../docs/modules/data/data_layer_new.md`](../../docs/modules/data/data_layer_new.md)。

## 概述

数据层提供统一的图数据集加载、序列化结果读取和BPE压缩管理接口。核心设计原则：

- **固定划分**：使用 `data.pkl + train/val/test_index.json`，不做随机划分
- **统一接口**：通过 `UnifiedDataInterface` 提供所有数据访问
- **显式构建**：数据预处理由独立脚本完成，接口只负责读取
- **零容错**：缺失文件直接报错，不使用fallback

## 核心组件

### 1. UnifiedDataInterface (UDI) - 统一数据接口

**位置**：`src/data/unified_data_interface.py`

**职责**：提供图数据、序列化结果、BPE压缩、词表等的统一读取接口

**主要方法**：

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

cfg = ProjectConfig()
udi = UnifiedDataInterface(cfg, dataset_name="qm9test")

# === 图数据读取 ===
graphs = udi.get_graphs()                    # 获取所有图
train_graphs = udi.get_graphs_by_split("train")  # 获取指定划分

# === 序列化结果读取 ===
# 方式1：所有数据（带图ID）
sequences, labels = udi.get_sequences(method="feuler")
# 返回: [(graph_id, [token_ids...]), ...], [{prop: value, ...}, ...]

# 方式2：按划分读取（带图ID）
train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = \
    udi.get_sequences_by_splits(method="feuler")

# 方式3：获取训练数据（带graph_id和完整属性字典）
(train_seqs, train_props), (val_seqs, val_props), (test_seqs, test_props) = \
    udi.get_training_data(method="feuler")
# 如需提取目标属性，需要手动处理：
# train_y = [prop['homo'] for prop in train_props]

# 方式4：获取扁平化序列（用于预训练）
train_seq, val_seq, test_seq = udi.get_training_data_flat(method="feuler")

# === 词表和BPE ===
vocab_manager = udi.get_vocab(method="feuler")  # 获取词表管理器
bpe_codebook = udi.get_bpe_codebook(method="feuler")  # 获取BPE码表

# === 元信息 ===
split_indices = udi.get_split_indices()      # 获取划分索引
metadata = udi.get_downstream_metadata()     # 获取下游任务元信息
```

### 2. BaseDataLoader - 数据加载器基类

**位置**：`src/data/base_loader.py`

**职责**：定义所有数据集加载器的统一接口

**必须实现的抽象方法**：

```python
class CustomLoader(BaseDataLoader):
    def _load_processed_data(self) -> Tuple[List, List, List]:
        """加载训练/验证/测试数据"""
        pass
    
    def _extract_labels(self, data: List) -> List:
        """从数据中提取标签"""
        pass
    
    def _get_data_metadata(self) -> Dict:
        """返回数据集元信息"""
        pass
    
    # Token管理接口
    def get_node_attribute(self, graph, node_id) -> int:
        """获取节点关键属性（用于token映射）"""
        pass
    
    def get_edge_attribute(self, graph, edge_id) -> int:
        """获取边关键属性（用于token映射）"""
        pass
    
    def get_node_type(self, graph, node_id) -> str:
        """获取节点类型名称"""
        pass
    
    def get_edge_type(self, graph, edge_id) -> str:
        """获取边类型名称"""
        pass
```

### 3. UnifiedDataFactory - 数据加载器工厂

**位置**：`src/data/unified_data_factory.py`

**职责**：管理所有数据集加载器的注册和创建

```python
from src.data.unified_data_factory import (
    get_dataloader,
    list_available_datasets,
    get_dataset_info
)

# 查看可用数据集
datasets = list_available_datasets()
print(datasets)  # ['qm9', 'qm9test', 'zinc', 'aqsol', 'mnist', ...]

# 创建加载器
loader = get_dataloader("qm9test", config)

# 查看数据集信息
info = get_dataset_info("qm9test")
```

## 已注册的数据集

当前支持的数据集（位于 `src/data/loader/` 目录）：

| 数据集 | 加载器文件 | 说明 |
|--------|-----------|------|
| `qm9` | `qm9_loader.py` | 完整QM9分子数据集 |
| `qm9test` | `qm9test_loader.py` | QM9测试子集（约10%） |
| `zinc` | `zinc_loader.py` | ZINC分子数据集 |
| `aqsol` | `aqsol_loader.py` | AQSOL溶解度数据集 |
| `mnist` | `mnist_loader.py` | MNIST超像素图数据集 |
| `mutagenicity` | `mutagenicity_loader.py` | 诱变性数据集 |
| `proteins` | `proteins_loader.py` | 蛋白质数据集 |
| `peptides_func` | `peptides_func_loader.py` | 肽功能预测 |
| `peptides_struct` | `peptides_struct_loader.py` | 肽结构预测 |
| `molhiv` | `molhiv_loader.py` | MoleculeNet HIV数据集 |
| `dd` | `dd_loader.py` | D&D蛋白质数据集 |
| `dblp` | `dblp_loader.py` | DBLP引用网络 |
| `code2` | `code2_loader.py` | CODE2数据集 |
| `coildel` | `coildel_loader.py` | COIL-DEL数据集 |
| `colors3` | `colors3_loader.py` | COLORS-3数据集 |
| `twitter` | `twitter_loader.py` | Twitter网络数据集 |
| `synthetic` | `synthetic_loader.py` | 合成图数据集 |

## 数据目录结构

每个数据集遵循统一的目录结构：

```
data/
├── <dataset_name>/
│   ├── data.pkl                  # 完整数据文件（必需）
│   ├── train_index.json          # 训练集索引（必需）
│   ├── val_index.json            # 验证集索引（必需）
│   ├── test_index.json           # 测试集索引（必需）
│   └── metadata.json             # 数据集元信息（可选）
│
└── processed/
    └── <dataset_name>/
        ├── serialized_data/
        │   └── <method>/         # 序列化方法（如feuler）
        │       └── single/       # 或 multi_N（多重采样）
        │           ├── serialized_data.pickle  # 序列化结果
        │           └── vocab.pkl               # 词表文件
        └── bpe_data/
            └── <method>/
                └── bpe_codebook.pkl  # BPE码表
```

## 完整使用示例

### 示例1：训练数据准备

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

# 1. 配置
cfg = ProjectConfig()
cfg.dataset.name = "qm9test"

# 2. 创建接口
udi = UnifiedDataInterface(cfg, cfg.dataset.name)

# 3. 获取训练数据
# 方式1：获取带graph_id的序列和完整属性字典
(train_seqs, train_props), (val_seqs, val_props), (test_seqs, test_props) = \
    udi.get_training_data(method="feuler")
# 提取目标属性
train_y = [prop['homo'] for prop in train_props]
val_y = [prop['homo'] for prop in val_props]
test_y = [prop['homo'] for prop in test_props]
# 提取序列（丢弃graph_id）
train_seq = [seq for _, seq in train_seqs]
val_seq = [seq for _, seq in val_seqs]
test_seq = [seq for _, seq in test_seqs]

# 方式2：直接获取扁平化序列（用于预训练，不需要属性）
train_seq, val_seq, test_seq = udi.get_training_data_flat(method="feuler")

print(f"训练集: {len(train_seq)} 样本")
print(f"验证集: {len(val_seq)} 样本")
print(f"测试集: {len(test_seq)} 样本")

# 4. 获取词表
vocab_manager = udi.get_vocab(method="feuler")
print(f"词表大小: {vocab_manager.vocab_size}")
```

### 示例2：序列化数据分析

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

cfg = ProjectConfig()
udi = UnifiedDataInterface(cfg, "qm9test")

# 获取序列化结果（带图ID）
sequences, labels = udi.get_sequences(method="feuler")

# 分析序列长度分布
lengths = [len(seq) for gid, seq in sequences]
print(f"平均长度: {sum(lengths)/len(lengths):.2f}")
print(f"最短: {min(lengths)}, 最长: {max(lengths)}")

# 查看标签结构
print(f"标签示例: {labels[0]}")  # {'homo': -0.5, 'lumo': 0.3, ...}
```

### 示例3：直接使用数据加载器

```python
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader

cfg = ProjectConfig()
loader = get_dataloader("qm9test", cfg)

# 获取划分数据
train_data, train_labels = loader.get_train_data()
val_data, val_labels = loader.get_val_data()
test_data, test_labels = loader.get_test_data()

# 获取Token映射
token_map = loader.get_token_map()
print(f"Token映射: {token_map}")

# 获取元信息
metadata = loader.get_data_metadata()
print(f"数据集信息: {metadata}")
```

## 错误处理

数据层采用**严格的错误处理策略**，所有异常情况都会明确抛出错误：

```python
# ❌ 数据集不存在
udi = UnifiedDataInterface(cfg, "unknown_dataset")
# → 抛出 FileNotFoundError: 数据集目录不存在

# ❌ 索引文件缺失
# → 抛出 FileNotFoundError: 训练集索引文件不存在

# ❌ 序列化结果未构建
sequences = udi.get_sequences(method="nonexistent_method")
# → 抛出 FileNotFoundError: 序列化结果不存在

# ✅ 正确处理
try:
    sequences = udi.get_sequences(method="feuler")
except FileNotFoundError as e:
    print(f"需要先运行数据预处理: {e}")
    # 运行 python prepare_data_new.py --dataset qm9test --method feuler
```

## 数据预处理

数据的序列化和BPE压缩由独立脚本完成：

```bash
# 准备序列化数据
python prepare_data_new.py \
    --dataset qm9test \
    --method feuler

# 如果需要BPE压缩
python prepare_data_new.py \
    --dataset qm9test \
    --method feuler \
    --enable-bpe \
    --num-merges 2000
```

UDI只负责读取已构建的数据，不进行构建。

## 设计原则说明

### 1. 为什么使用固定索引文件？

- **可重现性**：确保所有实验使用相同的数据划分
- **跨实验一致性**：不同方法、不同时间运行的实验结果可比较
- **透明性**：划分逻辑明确，不依赖隐藏的随机种子

### 2. 为什么读取接口不做数据构建？

- **职责分离**：数据预处理（耗时）和数据读取（快速）分离
- **明确依赖**：用户清楚知道需要先准备哪些数据
- **避免意外**：不会因为缓存逻辑导致使用了错误版本的数据

### 3. 为什么不使用fallback机制？

- **错误即停止**：及早发现数据问题，而不是用错误数据继续运行
- **配置明确性**：强制用户明确配置所需的数据
- **调试友好**：错误信息清晰，容易定位问题

## 扩展开发

### 添加新数据集

1. **创建加载器类**：

```python
# src/data/loader/my_dataset_loader.py
from src.data.base_loader import BaseDataLoader

class MyDatasetLoader(BaseDataLoader):
    def __init__(self, config, target_property=None):
        super().__init__("my_dataset", config, target_property)
    
    def _load_processed_data(self):
        # 实现数据加载逻辑
        pass
    
    def _extract_labels(self, data):
        # 实现标签提取逻辑
        pass
    
    def _get_data_metadata(self):
        # 返回数据集元信息
        pass
    
    def get_node_attribute(self, graph, node_id):
        # 实现节点属性获取
        pass
    
    def get_edge_attribute(self, graph, edge_id):
        # 实现边属性获取
        pass
    
    def get_node_type(self, graph, node_id):
        # 实现节点类型获取
        pass
    
    def get_edge_type(self, graph, edge_id):
        # 实现边类型获取
        pass
```

2. **注册到工厂**：

```python
# src/data/unified_data_factory.py 的 _lazy_import_loader 函数中添加
def _lazy_import_loader(loader_name: str):
    # ... 现有代码 ...
    elif loader_name == "my_dataset":
        from .loader.my_dataset_loader import MyDatasetLoader
        return MyDatasetLoader
    # ...

# 并在底部注册
UnifiedDataFactory.register("my_dataset", lambda: _lazy_import_loader("my_dataset"))
```

3. **准备数据文件**：

```bash
# 创建数据目录
mkdir -p data/my_dataset

# 放置必需文件
# - data/my_dataset/data.pkl
# - data/my_dataset/train_index.json
# - data/my_dataset/val_index.json
# - data/my_dataset/test_index.json
```

## 常见问题

### Q: 如何查看有哪些可用的数据集？

```python
from src.data.unified_data_factory import list_available_datasets
print(list_available_datasets())
```

### Q: 如何指定不同的序列化方法？

序列化方法在调用 `get_sequences()` 等方法时通过 `method` 参数指定。
可用方法见 `src/algorithms/serializer/` 目录。

### Q: 多属性数据集如何选择目标属性？

```python
# 方法1：获取完整属性字典，手动提取目标属性
train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = \
    udi.get_sequences_by_splits(method="feuler")
# train_labels 是字典列表，包含所有属性
train_y = [label['homo'] for label in train_labels]  # 提取homo属性

# 方法2：使用get_training_data获取后提取
(train_seqs, train_props), (val_seqs, val_props), (test_seqs, test_props) = \
    udi.get_training_data(method="feuler")
train_y = [prop['homo'] for prop in train_props]
```

### Q: 如何缓存数据以提高加载速度？

UDI支持预加载机制：

```python
udi = UnifiedDataInterface(cfg, "qm9test")
udi.preload_graphs()  # 预加载所有图数据到内存

# 后续操作会复用预加载的数据
sequences = udi.get_sequences(method="feuler")
```

## 相关文档

- [`../../docs/modules/data/data_layer_new.md`](../../docs/modules/data/data_layer_new.md) - 详细设计说明和演进历史
- [`../algorithms/serializer/README.md`](../algorithms/serializer/README.md) - 序列化算法文档
- [`../algorithms/compression/README.md`](../algorithms/compression/README.md) - BPE压缩文档
- [根目录`/docs/guides/coding_standards.md`](../../docs/guides/coding_standards.md) - 代码规范（数据契约章节）

---

**文档版本**：v2.0  
**最后更新**：2025-10-30  
**维护者**：请在发现文档与代码不符时立即提Issue或PR
