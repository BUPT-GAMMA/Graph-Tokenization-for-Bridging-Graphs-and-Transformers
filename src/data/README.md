# 数据层架构文档

> 重要：本页包含历史内容，部分接口已被新的统一数据接口替代。最新且准确的使用说明见 `src/data/DATA_LAYER_NEW.md`。
> 关键变化：
> - 统一通过 `UnifiedDataInterface` 读取图/序列/BPE，显式构建传 `build_if_missing=True`。
> - 固定划分采用三份索引 JSON，缺失即报错；不做隐式回退。
> - 历史接口如 `processed_data_loader.py`、`*dgl_loader.py` 不再推荐使用。

## 概述

本目录包含TokenizerGraph项目的数据层实现，提供从原始数据集到标准化训练数据源的完整流水线。

## 架构层次

### 1. 原始数据加载层
负责从各种原始数据源加载和预处理数据

| 文件 | 功能 | 使用场景 |
|------|------|----------|
| `qm9_loader.py` | 从QM9 CSV原始文件加载并构建DGL图 | 生产环境 |
| `qm9_dgl_loader.py` | 从预处理的DGL数据文件加载 | 生产环境 |
| `qm9_test_loader.py` | 10% QM9测试数据 | 开发/调试 |
| `qm9_test_dgl_loader.py` | 10% QM9 DGL测试数据 | 开发/调试 |

### 2. 统一接口层
提供标准化的数据访问接口

| 文件 | 功能 | 关键特性 |
|------|------|----------|
| `base_loader.py` | 所有加载器的抽象基类 | 统一接口定义 |
| `unified_data_factory.py` | 数据加载器工厂 | 注册和管理所有加载器 |
| `processed_data_loader.py` | **标准化数据源接口** | BERT训练数据入口 |

### 3. 辅助加载器
其他数据集的加载器（当前处于注释状态）

| 文件 | 状态 | 备注 |
|------|------|------|
| `loaders/` 目录下所有文件 | 已注册但注释掉 | 保留用于扩展 |

## 核心数据流程

### 生产流程
```
原始数据 → 数据加载器 → 序列化 → BPE压缩 → 标准化数据源 → 模型训练
    ↓        ↓           ↓        ↓           ↓           ↓
  QM9.csv → QM9Loader → graph_seq → BPE1000 → 预处理结果 → BERT训练
```

### 关键接口

#### 标准化数据源接口 (`processed_data_loader.py`)
```python
from src.data.processed_data_loader import load_final_data_source

# 加载BERT训练数据
data = load_final_data_source('qm9', 'graph_seq')
sequences = data['sequences']    # token序列
properties = data['properties']  # 分子属性
```

#### 统一数据工厂 (`unified_data_factory.py`)
```python
from src.data.unified_data_factory import get_dataloader

# 创建数据加载器
loader = get_dataloader('qm9', config)
data = loader.load_data(limit=1000)
```

## 注册的数据集

通过统一工厂注册的数据集：
- `qm9`: 完整QM9数据集
- `qm9_dgl`: 预处理的DGL格式QM9
- `qm9test`: 10% QM9测试数据
- `qm9test_dgl`: 10% DGL测试数据

## 目录结构

```
src/data/
├── README.md                    # 本文档
├── base_loader.py              # 抽象基类
├── unified_data_factory.py     # 统一数据工厂
├── processed_data_loader.py    # 标准化数据源接口
├── qm9_loader.py              # QM9原始数据加载器
├── qm9_dgl_loader.py          # QM9 DGL数据加载器
├── qm9_test_loader.py         # QM9测试数据加载器
├── qm9_test_dgl_loader.py     # QM9 DGL测试数据加载器
└── loaders/                   # 其他数据集加载器
    ├── [已注释的加载器]...
```

## 使用指南

### 开发阶段
```python
# 使用测试数据快速验证
from src.data.processed_data_loader import load_final_data_source
test_data = load_final_data_source('qm9test', 'graph_seq')
```

### 生产阶段
```python
# 使用完整数据源
full_data = load_final_data_source('qm9', 'graph_seq')
```

### 数据检查
```python
from src.data.processed_data_loader import list_available_datasets, list_available_methods

# 查看可用数据集
print(list_available_datasets())

# 查看可用序列化方法
print(list_available_methods('qm9'))
```

## 注意事项

1. **所有当前文件都在使用中** - 没有可以安全删除的文件
2. **processed_data_loader.py是关键接口** - BERT训练数据入口
3. **数据准备流程** - 使用根目录的 `data_prepare.py` 生成标准化数据源
4. **测试数据** - 开发时使用 `qm9test` 避免处理完整数据集

```python
from config import ProjectConfig
from src.data.unified_data_factory import get_dataset

config = ProjectConfig()

# 加载完整QM9 DGL数据集 (130,831个分子)
full_data = get_dataset('qm9_dgl', config)

# 加载QM9 DGL测试数据集 (10%数据，13,083个分子)
test_data = get_dataset('qm9test_dgl', config)

# 限制数据量
limited_data = get_dataset('qm9test_dgl', config, limit=100)
```

**QM9 DGL数据集特点：**
- **高效加载**: 直接从预处理的DGL数据文件加载
- **完整特征**: 包含14维节点特征和7维边特征
- **标准原子类型**: 包含C, H, O, N, F五种原子类型
- **3D坐标**: 包含分子的3D空间坐标信息
- **19个属性**: 包含完整的量子化学属性
- **统一接口**: 完全兼容BaseDataLoader接口

**数据格式：**
```python
sample = {
    'id': 'gdb_1',                # 分子ID
    'smiles': '[H]C([H])([H])[H]', # SMILES字符串
    'dgl_graph': dgl_graph,       # DGL图对象
    'num_nodes': 5,               # 节点数
    'num_edges': 4,               # 边数
    'dataset_name': 'qm9test_dgl', # 数据集名称
    'data_type': 'molecular_graph', # 数据类型
    'properties': {               # 19个量子化学属性
        'mu': 0.0,               # 偶极矩
        'alpha': 13.21,          # 极化率
        'homo': -10.55,          # HOMO能量
        'lumo': 3.19,            # LUMO能量
        'gap': 13.74,            # 能隙
        # ... 其他属性
    }
}
```

**特征详情：**
- **节点特征**: 14维向量 (原子类型one-hot编码[5] + 原子序数[1] + 其他特征[4] + 度数[1] + 3D坐标[3])
- **边特征**: 7维向量 (化学键类型one-hot编码[4] + 键长[1] + 键向量方向[2])
- **图元数据**: 包含分子量、转换方法、坐标等信息

详细文档请参考：[QM9 DGL数据加载器详细文档](QM9_DGL_Loader_Documentation.md)

#### 单图数据集

```python
from src.data.single_graph_loader import SingleGraphLoader

# 加载单个大图
loader = SingleGraphLoader("data/raw/cora")
graph = loader.load_graph()

# 创建子图样本
samples = SingleGraphLoader.create_full_graph_samples(
    graph, 
    num_samples=1000,
    strategy='random_walk'
)
```

### 1.3 数据加载器基类

如果需要创建自定义数据集加载器，继承 `BaseDataLoader`：

```python
from src.data.base_loader import BaseDataLoader
from typing import Dict, Any, List, Optional

class CustomLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__("custom_dataset", config)
    
    def _load_raw_data(self, limit: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """加载原始数据"""
        # 实现原始数据加载逻辑
        return raw_data
    
    def _process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理原始数据"""
        # 实现数据处理逻辑
        return processed_data
    
    def _get_data_metadata(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取数据元信息"""
        # 实现元信息生成逻辑
        return metadata
```

## 2. 预处理数据加载 (Stage Loaders)

### 2.1 序列化结果加载

#### 基本接口

```python
from src.data.processed_data_loader import load_serialization_result

# 加载序列化结果
result = load_serialization_result(
    dataset_name='qm9',           # 数据集名称
    method='graph_seq',           # 序列化方法
    run_id='run_001'              # 运行ID (可选)
)
```

#### 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `dataset_name` | str | 是 | 数据集名称，如 'qm9', 'cora' |
| `method` | str | 是 | 序列化方法，如 'graph_seq', 'dfs', 'bfs' |
| `run_id` | str | 否 | 运行ID，用于区分不同的实验运行 |

#### 返回格式

```python
{
    'sequences': [                # token序列列表
        [1, 2, 3, 4, 5],         # 第1个图的序列
        [6, 7, 8, 9, 10],        # 第2个图的序列
        # ...
    ],
    'metadata': {                 # 元数据
        'method': 'graph_seq',    # 序列化方法
        'dataset': 'qm9',         # 数据集名称
        'total_sequences': 1000,  # 总序列数
        'avg_length': 15.6,       # 平均长度
        'std_length': 3.2,        # 长度标准差
        'min_length': 8,          # 最小长度
        'max_length': 25,         # 最大长度
        'total_tokens': 15600,    # 总token数
        'success_rate': 0.98,     # 成功率
        'serialization_time': 45.2, # 序列化时间(秒)
        'serialization_speed': 22.1 # 序列化速度(图/秒)
    }
}
```

#### 使用示例

```python
# 加载QM9数据集的graph_seq序列化结果
result = load_serialization_result('qm9', 'graph_seq')

# 获取序列
sequences = result['sequences']
print(f"加载了 {len(sequences)} 个序列")

# 获取统计信息
metadata = result['metadata']
print(f"平均长度: {metadata['avg_length']:.2f}")

# 分析序列长度分布
lengths = [len(seq) for seq in sequences]
print(f"长度范围: {min(lengths)} - {max(lengths)}")
```

### 2.2 BPE模型加载

#### 基本接口

```python
from src.data.processed_data_loader import load_bpe_model

# 加载BPE模型
bpe_model = load_bpe_model(
    dataset_name='qm9',           # 数据集名称
    method='graph_seq',           # 序列化方法
    run_id='run_001'              # 运行ID (可选)
)
```

#### 返回对象

返回 `StandardBPECompressor` 实例，包含以下主要方法：

```python
# 编码序列
encoded = bpe_model.encode([1, 2, 3, 4, 5])

# 解码序列
decoded = bpe_model.decode([10, 20, 30])

# 获取词汇表大小
vocab_size = len(bpe_model.token_to_id)

# 获取合并规则
merge_rules = bpe_model.merge_rules

# 计算压缩统计
stats = bpe_model.calculate_compression_stats(sequences)
```

#### 使用示例

```python
# 加载BPE模型
bpe_model = load_bpe_model('qm9', 'graph_seq')

# 加载原始序列
serialization = load_serialization_result('qm9', 'graph_seq')
sequences = serialization['sequences']

# 测试压缩效果
test_sequences = sequences[:100]  # 取前100个序列测试
compressed = []

for seq in test_sequences:
    encoded = bpe_model.encode(seq)
    compressed.append(encoded)
    decoded = bpe_model.decode(encoded)
    assert decoded == seq  # 验证无损压缩

# 计算压缩统计
original_tokens = sum(len(seq) for seq in test_sequences)
compressed_tokens = sum(len(seq) for seq in compressed)
compression_ratio = compressed_tokens / original_tokens

print(f"压缩比: {compression_ratio:.3f}")
print(f"节省tokens: {original_tokens - compressed_tokens}")
```

### 2.3 BPE压缩结果加载

#### 基本接口

```python
from src.data.processed_data_loader import load_bpe_result

# 加载BPE压缩结果
result = load_bpe_result(
    dataset_name='qm9',           # 数据集名称
    method='graph_seq',           # 序列化方法
    run_id='run_001'              # 运行ID (可选)
)
```

#### 返回格式

```python
{
    'compressed_sequences': [     # 压缩后的序列
        [10, 20, 30],            # 第1个图的压缩序列
        [40, 50, 60],            # 第2个图的压缩序列
        # ...
    ],
    'compression_stats': {        # 压缩统计
        'compressed_total_tokens': 8000,    # 压缩后总token数
        'compressed_avg_length': 8.0,       # 压缩后平均长度
        'compression_ratio': 0.65,          # 压缩比
        'tokens_saved': 4200,               # 节省的token数
        'compression_percentage': 35.0      # 压缩率(%)
    },
    'model_info': {              # 模型信息
        'vocab_size': 5000,      # 词汇表大小
        'num_merges': 2000,      # 合并次数
        'min_frequency': 10,     # 最小频率
        'final_vocab_size': 5000 # 最终词汇表大小
    },
    'bpe_train_stats': {         # 训练统计
        'num_merges_performed': 2000,  # 实际执行合并次数
        'merge_rules_count': 2000,     # 合并规则数量
        'training_time': 120.5         # 训练时间(秒)
    },
    'decode_accuracy': 1.0,      # 解码准确率
    'sample_size': 1000          # 测试样本数
}
```

#### 使用示例

```python
# 加载BPE压缩结果
bpe_result = load_bpe_result('qm9', 'graph_seq')

# 获取压缩后的序列
compressed_sequences = bpe_result['compressed_sequences']
print(f"压缩后序列数: {len(compressed_sequences)}")

# 获取压缩统计
stats = bpe_result['compression_stats']
print(f"压缩比: {stats['compression_ratio']:.3f}")
print(f"节省tokens: {stats['tokens_saved']:,}")
print(f"压缩率: {stats['compression_percentage']:.1f}%")

# 获取模型信息
model_info = bpe_result['model_info']
print(f"词汇表大小: {model_info['vocab_size']:,}")
print(f"合并次数: {model_info['num_merges']}")

# 验证解码准确率
accuracy = bpe_result['decode_accuracy']
print(f"解码准确率: {accuracy:.1%}")
```

### 2.4 最终数据源加载

#### 基本接口

```python
from src.data.processed_data_loader import load_final_data_source

# 加载最终数据源
data_source = load_final_data_source(
    dataset_name='qm9',           # 数据集名称
    method='graph_seq',           # 序列化方法
    run_id='run_001'              # 运行ID (可选)
)
```

#### 返回格式

```python
{
    'train_data': {               # 训练数据
        'sequences': [...],       # 原始序列
        'compressed_sequences': [...], # 压缩序列
        'properties': [...],      # 图属性
        'indices': [...]          # 数据索引
    },
    'val_data': {                 # 验证数据
        'sequences': [...],
        'compressed_sequences': [...],
        'properties': [...],
        'indices': [...]
    },
    'test_data': {                # 测试数据
        'sequences': [...],
        'compressed_sequences': [...],
        'properties': [...],
        'indices': [...]
    },
    'vocab': {                    # 词汇表
        'token_to_id': {...},     # token到ID映射
        'id_to_token': {...},     # ID到token映射
        'vocab_size': 5000        # 词汇表大小
    },
    'config': {                   # 配置信息
        'dataset_name': 'qm9',
        'method': 'graph_seq',
        'bpe_config': {...},
        'split_ratio': [0.8, 0.1, 0.1]
    },
    'metadata': {                 # 元数据
        'total_samples': 1000,
        'train_samples': 800,
        'val_samples': 100,
        'test_samples': 100,
        'compression_ratio': 0.65,
        'avg_length': 15.6
    }
}
```

#### 使用示例

```python
# 加载最终数据源
data_source = load_final_data_source('qm9', 'graph_seq')

# 获取训练数据
train_sequences = data_source['train_data']['sequences']
train_compressed = data_source['train_data']['compressed_sequences']
train_properties = data_source['train_data']['properties']

print(f"训练样本数: {len(train_sequences)}")
print(f"验证样本数: {len(data_source['val_data']['sequences'])}")
print(f"测试样本数: {len(data_source['test_data']['sequences'])}")

# 获取词汇表
vocab = data_source['vocab']
print(f"词汇表大小: {vocab['vocab_size']}")

# 获取配置信息
config = data_source['config']
print(f"数据集: {config['dataset_name']}")
print(f"方法: {config['method']}")
print(f"BPE配置: {config['bpe_config']}")

# 用于机器学习训练
for i in range(len(train_sequences)):
    original_seq = train_sequences[i]
    compressed_seq = train_compressed[i]
    properties = train_properties[i]
    
    # 训练逻辑...
    pass
```

### 2.5 数据探索工具

```python
from src.data.processed_data_loader import (
    list_available_datasets,
    list_available_methods,
    get_dataset_info
)

# 查看可用的数据集
datasets = list_available_datasets()
print(f"可用数据集: {datasets}")

# 查看可用的处理方法
methods = list_available_methods('qm9')
print(f"QM9可用方法: {methods}")

# 获取数据集详细信息
info = get_dataset_info('qm9')
print(f"数据集信息: {info}")
```

## 3. 序列化方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| `'graph_seq'` | 图序列化 (推荐) | 通用图数据 |
| `'dfs'` | 深度优先搜索 | 树状结构 |
| `'bfs'` | 广度优先搜索 | 层次结构 |
| `'eulerian'` | 欧拉回路 | 连通图 |
| `'topological'` | 拓扑排序 | 有向无环图 |
| `'smiles'` | SMILES字符串 | 分子数据 |

## 4. 便捷函数

### 4.1 数据加载便捷函数

```python
from src.data import load_dataset, create_dataloader

# 快速加载数据
molecules = load_dataset('qm9', config, limit=1000)

# 创建加载器
loader = create_dataloader('qm9', config)
```

### 4.2 数据信息查询

```python
from src.data import list_available_datasets, get_dataset_info

# 查询可用数据集
datasets = list_available_datasets()

# 查询数据集信息
info = get_dataset_info('qm9')
```

## 5. 使用示例

### 5.1 完整的数据加载流程

```python
from config import ProjectConfig
from src.data.unified_data_factory import load_dataset
from src.algorithms.serializer import SerializerFactory

# 1. 配置
config = ProjectConfig()
config.subgraph_limt = 1000

# 2. 加载数据 (原始QM9数据集)
molecules = load_dataset('qm9', config, limit=1000)
print(f"加载了 {len(molecules)} 个分子")

# 或者加载QM9 DGL数据集 (推荐)
dgl_molecules = load_dataset('qm9_dgl', config, limit=1000)
print(f"加载了 {len(dgl_molecules)} 个DGL分子")

# 3. 创建序列化器
serializer = SerializerFactory.create_serializer('graph_seq')
serializer.initialize_with_dataset(molecules)  # 或 dgl_molecules

# 4. 序列化
sequences = []
for mol in molecules:  # 或 dgl_molecules
    result = serializer.serialize(mol)
    sequences.append(result.token_ids)

print(f"生成了 {len(sequences)} 个序列")
```

### 5.2 预处理数据使用

```python
from src.data.processed_data_loader import load_serialization_result, load_bpe_result

# 1. 加载序列化结果
serialization = load_serialization_result('qm9', 'graph_seq', 'run_001')
sequences = serialization['sequences']

# 2. 加载BPE结果
bpe_result = load_bpe_result('qm9', 'graph_seq', 'run_001')
compressed_sequences = bpe_result['compressed_sequences']

# 3. 使用数据
print(f"原始序列数: {len(sequences)}")
print(f"压缩后序列数: {len(compressed_sequences)}")
print(f"压缩比: {bpe_result['compression_stats']['compression_ratio']}")
```

### 5.3 数据验证和统计

```python
from src.data.unified_data_factory import load_dataset, get_dataset_info

# 1. 加载数据 (原始QM9数据集)
molecules = load_dataset('qm9', config, limit=100)

# 或者加载QM9 DGL数据集 (推荐)
dgl_molecules = load_dataset('qm9_dgl', config, limit=100)

# 2. 数据验证
for i, mol in enumerate(molecules):  # 或 dgl_molecules
    assert 'dgl_graph' in mol, f"样本 {i} 缺少DGL图"
    assert 'num_nodes' in mol, f"样本 {i} 缺少节点数"
    assert mol['num_nodes'] > 0, f"样本 {i} 节点数为0"

# 3. 统计信息
total_nodes = sum(mol['num_nodes'] for mol in molecules)  # 或 dgl_molecules
total_edges = sum(mol['num_edges'] for mol in molecules)  # 或 dgl_molecules
avg_nodes = total_nodes / len(molecules)  # 或 len(dgl_molecules)

print(f"总节点数: {total_nodes}")
print(f"总边数: {total_edges}")
print(f"平均节点数: {avg_nodes:.2f}")

# 4. DGL数据集特有验证
if 'qm9_dgl' in str(type(molecules)) or 'qm9test_dgl' in str(type(molecules)):
    # 验证节点特征维度
    sample_graph = molecules[0]['dgl_graph']
    node_feat_dim = sample_graph.ndata['feat'].shape[1]
    edge_feat_dim = sample_graph.edata['feat'].shape[1]
    print(f"节点特征维度: {node_feat_dim}")
    print(f"边特征维度: {edge_feat_dim}")
    
    # 验证分子属性
    properties = molecules[0]['properties']
    print(f"分子属性数量: {len(properties)}")
    print(f"属性示例: {list(properties.keys())[:5]}")
```

## 6. 配置说明

### 6.1 ProjectConfig 关键参数

```python
config = ProjectConfig()

# 数据量控制
config.subgraph_limt = 1000      # 限制加载的数据量
config.batch_size = 32           # 批处理大小

# 缓存设置
config.cache_dir = "data/cache"  # 缓存目录
config.use_cache = True          # 是否使用缓存

# 数据处理
config.num_workers = 4           # 多进程数量
config.device = "cuda"           # 设备选择
```

### 6.2 数据集特定配置

```python
# QM9数据集配置 (原始CSV格式)
config.qm9_data_dir = "data/raw/qm9"
config.qm9_properties = ["mu", "alpha", "homo", "lumo", "gap"]

# QM9 DGL数据集配置 (预处理的DGL格式，推荐)
config.qm9_dgl_data_dir = "data/qm9_dgl"
config.qm9_dgl_file = "qm9_dgl_full.pkl"
config.qm9test_dgl_ratio = 0.1  # 测试数据集比例

# 单图数据集配置
config.graph_data_dir = "data/raw/cora"
config.subgraph_strategy = "random_walk"
```

## 7. 错误处理

### 7.1 常见错误及解决方案

```python
# 1. 数据集不存在
try:
    molecules = load_dataset('unknown_dataset', config)
except ValueError as e:
    print(f"数据集不存在: {e}")
    # 查看可用数据集
    available = list_available_datasets()
    print(f"可用数据集: {available}")

# 2. 缓存文件损坏
try:
    molecules = load_dataset('qm9', config)
except Exception as e:
    print(f"缓存错误: {e}")
    # 清除缓存重新加载
    config.use_cache = False
    molecules = load_dataset('qm9', config)

# 3. 数据格式错误
try:
    for mol in molecules:
        graph = mol['dgl_graph']
        # 验证图结构
        assert graph.num_nodes() > 0
except AssertionError:
    print("数据格式错误，请检查数据源")

# 4. 序列化方法不存在
try:
    result = load_serialization_result('qm9', 'unknown_method')
except FileNotFoundError as e:
    print(f"序列化方法不存在: {e}")
    # 查看可用方法
    methods = list_available_methods('qm9')
    print(f"可用方法: {methods}")
```

### 7.2 数据验证

```python
def validate_serialization_result(result):
    """验证序列化结果"""
    assert 'sequences' in result, "缺少sequences字段"
    assert 'metadata' in result, "缺少metadata字段"
    
    sequences = result['sequences']
    metadata = result['metadata']
    
    # 验证序列数量
    assert len(sequences) == metadata['total_sequences'], "序列数量不匹配"
    
    # 验证序列长度
    lengths = [len(seq) for seq in sequences]
    assert abs(sum(lengths) / len(lengths) - metadata['avg_length']) < 0.1, "平均长度不匹配"
    
    print("✅ 序列化结果验证通过")

def validate_bpe_result(result):
    """验证BPE结果"""
    assert 'compressed_sequences' in result, "缺少compressed_sequences字段"
    assert 'compression_stats' in result, "缺少compression_stats字段"
    
    compressed = result['compressed_sequences']
    stats = result['compression_stats']
    
    # 验证压缩比
    total_compressed = sum(len(seq) for seq in compressed)
    assert total_compressed == stats['compressed_total_tokens'], "压缩后token数不匹配"
    
    print("✅ BPE结果验证通过")
```

## 8. 性能优化

### 8.1 缓存策略

```python
# 启用缓存 (推荐)
config.use_cache = True
config.cache_dir = "data/cache"

# 首次加载会较慢，后续加载会很快
molecules = load_dataset('qm9', config, limit=1000)

# QM9 DGL数据集缓存效果更明显
dgl_molecules = load_dataset('qm9_dgl', config, limit=1000)
```

### 8.2 内存优化

```python
# 分批加载大数据集
batch_size = 1000
total_molecules = []

for i in range(0, 10000, batch_size):
    batch = load_dataset('qm9', config, limit=batch_size, offset=i)
    total_molecules.extend(batch)
    print(f"已加载 {len(total_molecules)} 个分子")

# QM9 DGL数据集分批加载
total_dgl_molecules = []
for i in range(0, 10000, batch_size):
    batch = load_dataset('qm9_dgl', config, limit=batch_size, offset=i)
    total_dgl_molecules.extend(batch)
    print(f"已加载 {len(total_dgl_molecules)} 个DGL分子")
```

### 8.3 并行处理

```python
# 使用多进程
config.num_workers = 4
config.use_cache = True

# 并行加载多个数据集
from concurrent.futures import ThreadPoolExecutor

def load_dataset_parallel(dataset_name):
    return load_dataset(dataset_name, config, limit=100)

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(load_dataset_parallel, ['qm9', 'qm9_dgl', 'cora']))
```

### 8.4 按需加载

```python
# 只加载需要的数据
if need_sequences_only:
    result = load_serialization_result('qm9', 'graph_seq')
    sequences = result['sequences']
elif need_compressed_only:
    result = load_bpe_result('qm9', 'graph_seq')
    compressed = result['compressed_sequences']
elif need_full_data:
    data_source = load_final_data_source('qm9', 'graph_seq')
elif need_dgl_data:
    # 直接加载DGL数据，无需序列化
    dgl_data = load_dataset('qm9_dgl', config, limit=1000)
```

## 9. 使用场景

### 9.1 序列分析

```python
# 分析序列特征
result = load_serialization_result('qm9', 'graph_seq')
sequences = result['sequences']

# 长度分布分析
lengths = [len(seq) for seq in sequences]
print(f"长度统计: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.2f}")

# 词汇使用分析
all_tokens = []
for seq in sequences:
    all_tokens.extend(seq)
token_counts = Counter(all_tokens)
print(f"词汇使用统计: {len(token_counts)} 个唯一token")
```

### 9.2 压缩效果分析

```python
# 比较不同方法的压缩效果
methods = ['graph_seq', 'dfs', 'bfs', 'eulerian']

for method in methods:
    bpe_result = load_bpe_result('qm9', method)
    stats = bpe_result['compression_stats']
    print(f"{method}: 压缩比={stats['compression_ratio']:.3f}, "
          f"节省={stats['tokens_saved']:,} tokens")
```

### 9.3 机器学习训练

```python
# 准备训练数据
data_source = load_final_data_source('qm9', 'graph_seq')

train_sequences = data_source['train_data']['compressed_sequences']
train_properties = data_source['train_data']['properties']
vocab = data_source['vocab']

# 训练模型
for epoch in range(num_epochs):
    for i, (seq, props) in enumerate(zip(train_sequences, train_properties)):
        # 训练逻辑
        pass
```

## 10. 扩展开发

### 10.1 添加新数据集

```python
from src.data.base_loader import BaseDataLoader
from src.data.unified_data_factory import UnifiedDataFactory

class NewDatasetLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__("new_dataset", config)
    
    def _load_raw_data(self, limit=None, **kwargs):
        # 实现原始数据加载
        pass
    
    def _process_raw_data(self, raw_data):
        # 实现数据处理
        pass
    
    def _get_data_metadata(self, processed_data):
        # 实现元信息生成
        pass

# 注册新数据集
UnifiedDataFactory.register('new_dataset', NewDatasetLoader)
```

### 10.2 自定义数据处理器

```python
class CustomDataProcessor:
    def __init__(self, config):
        self.config = config
    
    def process(self, raw_data):
        # 实现自定义处理逻辑
        return processed_data
    
    def validate(self, data):
        # 实现数据验证逻辑
        return is_valid
```

## 11. 快速参考

### 11.1 常用接口速查

```python
# 原生数据加载
molecules = load_dataset('qm9', config, limit=1000)

# QM9 DGL数据加载 (推荐)
dgl_molecules = load_dataset('qm9_dgl', config, limit=1000)
test_molecules = load_dataset('qm9test_dgl', config, limit=100)

# 序列化结果
result = load_serialization_result('qm9', 'graph_seq')
sequences = result['sequences']

# BPE模型
bpe_model = load_bpe_model('qm9', 'graph_seq')
encoded = bpe_model.encode([1, 2, 3, 4, 5])

# BPE结果
result = load_bpe_result('qm9', 'graph_seq')
compressed = result['compressed_sequences']

# 最终数据源
data_source = load_final_data_source('qm9', 'graph_seq')
train_data = data_source['train_data']
```

### 11.2 支持的数据集

- `'qm9'` - 分子数据集 (原始CSV格式)
- `'qm9_dgl'` - QM9 DGL数据集 (预处理的DGL格式，推荐)
- `'qm9test_dgl'` - QM9 DGL测试数据集 (10%数据)
- `'cora'` - 引文网络
- `'citeseer'` - 引文网络
- `'pubmed'` - 引文网络
- `'dblp'` - 异构网络
- `'imdb'` - 电影网络
- `'lastfm'` - 音乐网络
- `'yelp'` - 商业网络

### 11.3 配置参数

```python
config = ProjectConfig()

# 数据量控制
config.subgraph_limt = 1000      # 限制数据量
config.batch_size = 32           # 批处理大小

# 缓存设置
config.cache_dir = "data/cache"  # 缓存目录
config.use_cache = True          # 使用缓存

# 数据处理
config.num_workers = 4           # 多进程数
config.device = "cuda"           # 设备
```

## 12. 最佳实践

1. **使用统一接口**: 优先使用 `load_dataset()` 函数
2. **启用缓存**: 提高重复加载的效率
3. **按需加载**: 避免一次性加载过多数据
4. **错误处理**: 添加适当的异常处理
5. **数据验证**: 验证加载数据的完整性
6. **性能监控**: 监控加载时间和内存使用

## 总结

数据层模块提供了完整的数据加载和处理解决方案：

1. **统一接口**: 所有数据集使用相同的接口
2. **灵活配置**: 支持多种配置选项
3. **缓存机制**: 提高加载效率
4. **错误处理**: 完善的错误处理机制
5. **扩展性**: 易于添加新数据集
6. **按需加载**: 支持加载不同阶段的数据

推荐使用 `load_dataset()` 函数进行数据加载，这是最简单和最直接的方式。对于特殊需求，可以使用 `create_dataloader()` 创建加载器实例进行更精细的控制。 