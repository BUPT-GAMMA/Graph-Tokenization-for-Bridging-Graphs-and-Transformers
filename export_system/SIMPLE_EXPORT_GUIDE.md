# 简单数据集导出指南

## 📋 概述

这是一个**极简**的数据集导出方案说明。这里描述的是推荐接口，不表示仓库当前已经包含每个数据集对应的独立导出脚本文件。

## 🎯 设计原则

1. **一个数据集一个脚本**：推荐形态是每个数据集有独立的 `export_<dataset>.py`
2. **极简数据格式**：只保存构建图需要的基本信息
3. **直接可用**：导出的数据可以直接转换为DGL或PyG格式
4. **无复杂逻辑**：没有条件判断、假设、兼容性代码

## 📁 文件结构

```
项目根目录/
├── simple_graph_loader.py     # 通用加载器
├── export_<dataset>.py       # 计划中的单数据集导出脚本
└── export_<dataset>.py       # 其他数据集导出脚本
```

## 📊 数据格式

### 统一格式定义

```python
{
    'graphs': [
        {
            'src': [0, 1, 2, ...],           # 源节点ID列表
            'dst': [1, 2, 0, ...],           # 目标节点ID列表  
            'num_nodes': int,                # 节点总数
            'node_feat': [[f1], [f2], ...],  # 节点特征 [N, node_dim]
            'edge_feat': [[f1], [f2], ...],  # 边特征 [E, edge_dim]
        },
        ...
    ],
    'labels': [label1, label2, ...],       # 标签列表
    'splits': {                            # 数据划分索引
        'train': [0, 1, 2, ...],
        'val': [100, 101, ...],
        'test': [200, 201, ...]
    }
}
```

### 特征说明

- **节点特征**：通常是原子序数或离散token值
- **边特征**：通常是键类型或距离值  
- **标签**：
  - 分类任务：整数类别 `1, 2, 3`
  - 回归任务：浮点数值 `0.123`
  - 多标签：字典 `{'mu': 0.1, 'alpha': 2.3}`

## 🚀 使用方法

### 第1步：导出数据集

```bash
# 按约定形态调用
python export_<dataset>.py
# 输出: <dataset>_simple.pkl
```

### 第2步：在目标项目中使用

将 `simple_graph_loader.py` 和数据文件复制到目标项目：

```python
from simple_graph_loader import *

# 加载数据
data = load_simple_graph_data('qm9_simple.pkl')
print(f"加载了 {len(data['graphs'])} 个图")

# 转换为DGL格式
dgl_data = to_dgl(data)
for graph, label in dgl_data[:5]:
    print(f"节点数: {graph.num_nodes()}, 边数: {graph.num_edges()}")

# 转换为PyG格式  
pyg_data = to_pyg(data)
for data_obj in pyg_data[:5]:
    print(f"节点特征: {data_obj.x.shape}, 边: {data_obj.edge_index.shape}")

# 获取训练集数据
train_graphs, train_labels = get_split_data(data, 'train')
```

## 📝 为新数据集创建导出脚本

### 模板脚本

```python
#!/usr/bin/env python3
"""
<DATASET_NAME>数据集导出脚本
"""

import pickle
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader


def export_<dataset_name>(output_file: str = "<dataset_name>_simple.pkl"):
    """导出<DATASET_NAME>数据集"""
    
    print("🔄 开始导出<DATASET_NAME>数据集...")
    
    # 加载数据集
    config = ProjectConfig()
    loader = get_dataloader("<dataset_name>", config)
    all_data, split_indices = loader.get_all_data_with_indices()
    
    graphs = []
    labels = []
    
    for sample in all_data:
        dgl_graph = sample['dgl_graph']
        properties = sample['properties']
        
        # 提取图结构
        src, dst = dgl_graph.edges()
        
        # 提取特征（根据数据集具体格式）
        # 节点特征：从 ndata['feat'] 或 ndata['attr'] 等提取
        # 边特征：从 edata['feat'] 或 edata['edge_attr'] 等提取
        
        simple_graph = {
            'src': src.tolist(),
            'dst': dst.tolist(),
            'num_nodes': int(dgl_graph.num_nodes()),
            'node_feat': node_feat,  # 根据具体情况提取
            'edge_feat': edge_feat,  # 根据具体情况提取
        }
        
        graphs.append(simple_graph)
        labels.append(label)  # 根据具体情况提取标签
    
    # 保存数据
    simple_data = {'graphs': graphs, 'labels': labels, 'splits': split_indices}
    with open(output_file, 'wb') as f:
        pickle.dump(simple_data, f)
    
    print(f"✅ 导出完成: {output_file}")


if __name__ == "__main__":
    export_<dataset_name>()
```

### 具体示例

#### 分子数据集（原子序数 + 键类型）

```python
# 节点特征：原子序数
if 'feat' in dgl_graph.ndata:
    node_data = dgl_graph.ndata['feat']  # [N] 或 [N, 1]
    node_feat = [[float(x)] for x in node_data.tolist()]

# 边特征：键类型  
if 'feat' in dgl_graph.edata:
    edge_data = dgl_graph.edata['feat']  # [E] 或 [E, 1]
    edge_feat = [[float(x)] for x in edge_data.tolist()]
```

#### 图分类数据集（离散token）

```python
# 节点特征：离散token
if 'node_token_ids' in dgl_graph.ndata:
    tokens = dgl_graph.ndata['node_token_ids']
    node_feat = [[float(x)] for x in tokens.view(-1).tolist()]

# 边特征：常数或离散值
if 'edge_token_ids' in dgl_graph.edata:
    tokens = dgl_graph.edata['edge_token_ids']  
    edge_feat = [[float(x)] for x in tokens.view(-1).tolist()]
else:
    edge_feat = [[0.0] for _ in range(len(src))]  # 常数填充
```

#### 图像数据集（多维特征）

```python
# 节点特征：多维（如MNIST的pixel, x, y坐标）
if 'feature' in dgl_graph.ndata:
    features = dgl_graph.ndata['feature']  # [N, 3]
    node_feat = [[float(x) for x in row] for row in features.tolist()]
```

## ✅ 已实现的数据集

| 数据集 | 计划中的脚本形态 | 节点特征 | 边特征 | 标签类型 |
|--------|----------|----------|---------|----------|
| QM9 | `export_<dataset>.py` | 原子序数(1维) | 键类型(1维) | 多属性回归 |
| ZINC | `export_<dataset>.py` | 原子序数(1维) | 键类型(1维) | 单值回归 |
| MOLHIV | `export_<dataset>.py` | 原子序数(1维) | 键类型(1维) | 二分类 |

## 🔧 在其他项目中集成

### PyG项目示例

```python
# dataset.py
from torch_geometric.data import Dataset
from simple_graph_loader import load_simple_graph_data, to_pyg, get_split_data

class SimpleDataset(Dataset):
    def __init__(self, data_file, split='train'):
        super().__init__()
        self.data = load_simple_graph_data(data_file)
        self.graphs, self.labels = get_split_data(self.data, split)
        
    def len(self):
        return len(self.graphs)
        
    def get(self, idx):
        return to_pyg({'graphs': [self.graphs[idx]], 'labels': [self.labels[idx]]})[0]

# 使用
dataset = SimpleDataset('qm9_simple.pkl', 'train')
loader = DataLoader(dataset, batch_size=32)
```

### DGL项目示例

```python
# dataset.py
from simple_graph_loader import load_simple_graph_data, to_dgl, get_split_data

def get_dgl_dataloader(data_file, split='train', batch_size=32):
    data = load_simple_graph_data(data_file)
    graphs, labels = get_split_data(data, split)
    dgl_data = to_dgl({'graphs': graphs, 'labels': labels})
    
    # 分离图和标签
    dgl_graphs = [g for g, l in dgl_data]
    return dgl.dataloading.GraphDataLoader(dgl_graphs, batch_size=batch_size)

# 使用
train_loader = get_dgl_dataloader('molhiv_simple.pkl', 'train')
```

## 🎯 总结

这个方案极其简单：

1. **每个数据集一个脚本**：清晰，无依赖
2. **统一简单格式**：只有图结构 + 特征 + 标签
3. **直接可用**：一个函数调用就能转换为DGL/PyG
4. **易于扩展**：复制模板，填写特征提取逻辑即可

完美适合跨项目数据共享！
