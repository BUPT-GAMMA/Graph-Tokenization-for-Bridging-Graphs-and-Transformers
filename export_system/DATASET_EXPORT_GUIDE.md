# 数据集简单导出与跨项目使用指南

## 📋 概述

本指南提供了将项目中的DGL格式数据集导出为**极简格式**的方案。文中的脚本命名是设计接口，不表示仓库当前已经包含按数据集拆分好的独立导出脚本。

## 🎯 核心优势

1. **极简设计**：每个数据集一个独立脚本，无复杂逻辑
2. **直接可用**：导出的数据直接转换为DGL或PyG格式
3. **最小依赖**：只需要torch和相应图库
4. **易于理解**：清晰的数据格式和转换逻辑

## 🏗️ 方案架构

```
原始项目数据集              简单导出格式              目标项目使用
├── data/qm9/              ├── qm9_simple.pkl        ├── simple_graph_loader.py
├── data/zinc/             ├── zinc_simple.pkl       └── your_model.py
├── data/molhiv/           ├── molhiv_simple.pkl
└── ...                   └── ...
```

## 🚀 使用步骤

### 第一步：导出数据集

若后续补齐独立导出脚本，可按下面的接口模式运行；当前仓库并未提供这些按数据集拆分的独立脚本文件：

```bash
# python export_<dataset>.py
```

### 第二步：移植到目标项目

将数据文件和加载器复制到目标项目：

```bash
# 复制数据文件和加载器
cp qm9_simple.pkl /path/to/target/project/
cp simple_graph_loader.py /path/to/target/project/
```

### 第三步：在目标项目中使用

```python
from simple_graph_loader import *

# 加载数据
data = load_simple_graph_data('qm9_simple.pkl')

# 转换为PyG格式
pyg_data = to_pyg(data)

# 转换为DGL格式
dgl_data = to_dgl(data)

# 获取训练集
train_graphs, train_labels = get_split_data(data, 'train')
```

## 📊 数据格式说明

### 极简数据格式

```python
{
    'graphs': [
        {
            'src': [0, 1, 2, ...],           # 源节点列表
            'dst': [1, 2, 0, ...],           # 目标节点列表
            'num_nodes': int,                # 节点总数
            'node_feat': [[f1], [f2], ...],  # 节点特征 [N, 1]
            'edge_feat': [[f1], [f2], ...],  # 边特征 [E, 1]
        },
        ...
    ],
    'labels': [label1, label2, ...],       # 标签列表
    'splits': {                            # 数据划分
        'train': [0, 1, 2, ...],
        'val': [100, 101, ...],
        'test': [200, 201, ...]
    }
}
```

### 已实现的数据集

| 数据集 | 计划中的脚本形态 | 节点特征 | 边特征 | 标签 |
|--------|----------|----------|---------|-------|
| QM9 | `export_<dataset>.py` | 原子序数 | 键类型 | 分子属性字典 |
| ZINC | `export_<dataset>.py` | 原子序数 | 键类型 | 回归数值 |
| MOLHIV | `export_<dataset>.py` | 原子序数 | 键类型 | 分类标签 |

## 📝 为新数据集创建导出脚本

复制现有脚本模板，修改特征提取逻辑：

```python
#!/usr/bin/env python3
"""
YOUR_DATASET数据集导出脚本
"""

import pickle
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader


def export_your_dataset(output_file: str = "your_dataset_simple.pkl"):
    config = ProjectConfig()
    loader = get_dataloader("your_dataset", config)
    all_data, split_indices = loader.get_all_data_with_indices()
    
    graphs = []
    labels = []
    
    for sample in all_data:
        dgl_graph = sample['dgl_graph']
        properties = sample['properties']
        
        # 提取图结构
        src, dst = dgl_graph.edges()
        
        # 根据具体数据集提取特征
        # node_feat = [...] 
        # edge_feat = [...]
        
        simple_graph = {
            'src': src.tolist(),
            'dst': dst.tolist(), 
            'num_nodes': int(dgl_graph.num_nodes()),
            'node_feat': node_feat,
            'edge_feat': edge_feat,
        }
        
        graphs.append(simple_graph)
        labels.append(label)  # 根据任务类型提取
    
    simple_data = {'graphs': graphs, 'labels': labels, 'splits': split_indices}
    with open(output_file, 'wb') as f:
        pickle.dump(simple_data, f)

if __name__ == "__main__":
    export_your_dataset()
```

## 🧪 测试验证

```bash
# 测试导出的数据
python test_simple_export.py

# 输出示例：
# 🔍 测试 qm9_simple.pkl
# ✅ 数据加载成功: 130831 个图
#    📊 样本图: 29 节点, 64 条边
#    🎯 节点特征维度: 1
#    🎯 边特征维度: 1
#    ✅ DGL转换成功: 5 个图
#    ✅ PyG转换成功: 5 个图
```

---

通过这套**极简**的导出方案，您可以轻松地将我们的预处理数据移植到任何项目中！
