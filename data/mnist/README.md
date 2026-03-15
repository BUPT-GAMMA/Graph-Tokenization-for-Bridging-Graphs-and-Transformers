# MNIST图数据转换

这个目录包含了将MNIST数据集转换为图格式的工具和脚本。

## 文件说明

- `final_slic.py`: 核心的图像到图转换函数，使用SLIC超像素分割算法
- `convert_mnist_to_dgl.py`: 主转换脚本，将MNIST图像转换为DGL图格式
- `usage_example.py`: 使用示例脚本，展示如何加载和使用转换后的数据
- `README.md`: 本说明文件

## 数据格式

转换后的数据包含以下文件：

- `data.pkl`: 包含所有DGL图对象的pickle文件
- `train_index.json`: 训练集索引列表
- `val_index.json`: 验证集索引列表  
- `test_index.json`: 测试集索引列表
- `conversion_stats.json`: 转换统计信息

## 使用方法

### 1. 转换MNIST数据为图格式

```bash
cd data/mnist
python convert_mnist_to_dgl.py
```

默认配置：
- 处理10000个样本
- 训练集:验证集:测试集 = 8:1:1
- 随机种子: 42

### 2. 加载和使用转换后的数据

```python
from usage_example import load_mnist_graph_data, get_graph_features

# 加载数据
graphs, train_indices, val_indices, test_indices = load_mnist_graph_data()

# 获取图特征
node_features, edge_features, label = get_graph_features(graphs[0])
```

### 3. 运行使用示例

```bash
python usage_example.py
```

## 图结构

每个图包含以下信息：

- **节点特征**: 3维向量 [pixel_id, y_id, x_id]
  - pixel_id: 像素值分箱 (0-255)
  - y_id: Y坐标分箱 (0-27)
  - x_id: X坐标分箱 (0-27)

- **边特征**: 1维向量 [distance_id]
  - distance_id: 距离分箱 (0-39)

- **图标签**: 数字标签 (0-9)

## 配置参数

在 `final_slic.py` 中的 `GRAPH_PARAMS` 可以调整：

```python
GRAPH_PARAMS = {
    'pixel_bins': 256,      # 像素值分箱数量
    'y_bins': 28,           # Y坐标分箱数量
    'x_bins': 28,           # X坐标分箱数量
    'distance_bins': 40,    # 距离分箱数量
    'max_distance': 40,     # 最大距离
    'threshold': 10,        # 像素阈值
    'n_segments_digit': 50, # 数字区域超像素数量
    'n_segments_bg': 20,    # 背景区域超像素数量
}
```

## 依赖库

- numpy
- networkx
- dgl
- torch
- tensorflow
- scikit-learn
- matplotlib
- scikit-image

## 注意事项

1. 转换过程可能需要较长时间，建议先用少量样本测试
2. 确保有足够的内存来存储转换后的图数据
3. 转换失败的情况会被记录并跳过
4. 数据集划分使用分层采样，确保每个数字类别都有代表性 