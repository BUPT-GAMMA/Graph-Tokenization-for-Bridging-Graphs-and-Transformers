# MNIST图数据集文档

## 数据集概述

本数据集将MNIST手写数字图像转换为图格式，使用SLIC超像素分割算法生成图结构，保存为DGL格式。

## 数据集统计

- 总样本数: 70,000个图
- 训练集: 55,999个图 (80%)
- 验证集: 7,001个图 (10%)
- 测试集: 7,000个图 (10%)
- 类别数: 10个数字类别 (0-9)
- 数据大小: 783MB

### 标签分布

| 数字 | 训练集 | 验证集 | 测试集 | 总计 |
|------|--------|--------|--------|------|
| 0    | 6,903  | 863    | 863    | 8,629 |
| 1    | 7,877  | 985    | 985    | 9,847 |
| 2    | 6,990  | 874    | 874    | 8,738 |
| 3    | 7,141  | 893    | 893    | 8,927 |
| 4    | 6,824  | 853    | 853    | 8,530 |
| 5    | 6,313  | 789    | 789    | 7,891 |
| 6    | 6,876  | 860    | 860    | 8,596 |
| 7    | 7,293  | 912    | 912    | 9,117 |
| 8    | 6,825  | 853    | 853    | 8,531 |
| 9    | 6,958  | 870    | 870    | 8,698 |

## 文件结构

```
data/mnist/
├── data.pkl                    # 主要数据文件
├── train_index.json           # 训练集索引
├── val_index.json             # 验证集索引
├── test_index.json            # 测试集索引
├── conversion_stats.json      # 转换统计信息
├── convert_mnist_to_dgl.py    # 转换脚本
├── usage_example.py           # 使用示例
└── final_slic.py              # SLIC算法实现
```

## 数据格式

### 主要数据文件 (data.pkl)

包含一个列表，每个元素为`(graph, label)`元组：

```python
[
    (dgl_graph_0, label_0),
    (dgl_graph_1, label_1),
    ...
    (dgl_graph_69999, label_69999)
]
```

- `dgl_graph_i`: DGL图对象
- `label_i`: 整数标签 (0-9)

### 索引文件

JSON格式的整数数组，表示样本在data.pkl中的索引位置：

```json
[20067, 56390, 49027, 65443, ...]
```

## 图结构

### 节点特征

每个节点包含3维特征向量：
- 维度0: 像素值 (0-255)
- 维度1: Y坐标 (0-27)  
- 维度2: X坐标 (0-27)

### 边特征

每条边包含1维特征：
- 距离分箱 (0-39)

### 图属性

- 有向图
- 节点数: 62~75
- 边数: 140~390

## 转换参数

```python
GRAPH_PARAMS = {
    'pixel_bins': 256,      # 像素值分箱数
    'y_bins': 28,           # Y坐标分箱数
    'x_bins': 28,           # X坐标分箱数
    'distance_bins': 40,    # 距离分箱数
    'max_distance': 40,     # 最大距离
    'threshold': 10,        # 像素阈值
    'n_segments_digit': 50, # 数字区域超像素数
    'n_segments_bg': 20,    # 背景区域超像素数
}
```

## 使用方法

### 加载数据

```python
import pickle
import json

# 加载图数据
with open('data/mnist/data.pkl', 'rb') as f:
    graph_label_pairs = pickle.load(f)

# 加载索引
with open('data/mnist/train_index.json', 'r') as f:
    train_indices = json.load(f)
```

### 访问样本

```python
# 获取第一个训练样本
train_idx = train_indices[0]
graph, label = graph_label_pairs[train_idx]

# 获取特征
node_features = graph.ndata['feature']  # 形状: (num_nodes, 3)
edge_features = graph.edata['feature']  # 形状: (num_edges, 1)
```

### 创建数据加载器

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MNISTGraphDataset(Dataset):
    def __init__(self, graph_label_pairs, indices):
        self.graph_label_pairs = graph_label_pairs
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        graph, label = self.graph_label_pairs[data_idx]
        return graph, torch.tensor(label, dtype=torch.long)

# 创建数据集
train_dataset = MNISTGraphDataset(graph_label_pairs, train_indices)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## 转换过程

1. 图像预处理：阈值分割区分数字和背景
2. SLIC分割：数字区域50个超像素，背景区域20个超像素
3. 特征提取：计算节点和边的离散化特征
4. 图构建：基于空间邻接关系建立连接

## 注意事项

- 图规模因图像内容而异
- 所有特征经过离散化处理
- 使用有向边表示
- 需要足够内存加载完整数据集 