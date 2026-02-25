# 效率对比实验使用说明

## 🎯 实验目标

本实验用于生成论文中的效率对比图表，包括：
1. **序列化长度对比** - 展示不同序列化方法的序列长度和BPE压缩效果
2. **序列化速度对比** - 展示不同序列化方法的处理速度
3. **训练效率对比** - 展示序列化方法与传统图神经网络的训练效率对比

## 🚀 快速开始

### 方法一：一键生成所有图表
```bash
cd /home/gzy/py/tokenizerGraph/final/exp1_speed
python run_all_plots.py
```

如果想同时显示图表：
```bash
python run_all_plots.py --show
```

### 方法二：分别生成各类图表

#### 1. 序列化长度对比
```bash
cd token_length/
python plot_token_length.py
```

#### 2. 序列化速度对比
```bash
cd serialize_time/
python plot_serialize_speed.py
```

#### 3. 训练效率对比
```bash
cd train_time/
python plot_train_efficiency.py
```

## 📊 数据格式说明

### 序列化长度数据 (token_length/*.csv)
```csv
serialization_method,original_length,bpe_compressed_length,compression_rate
dfs,145.2,15.3,0.105
bfs,142.8,14.8,0.104
...
```
**注**: 序列化方法名称基于src/algorithms/serializer/serializer_factory.py定义

### 序列化速度数据 (serialize_time/*.csv)
```csv
serialization_method,graphs_per_second
dfs,1250.3
bfs,1180.7
...
```

### 训练效率数据 (train_time/*.csv)
```csv
method_name,epoch_time_seconds,method_type,method_group
dfs,45.2,serialization_no_bpe,dfs
dfs_bpe,38.7,serialization_with_bpe,dfs
GCN,89.4,graph_neural_network,gnn
...
```
**注**: 新增method_group列用于将w和w/o BPE的柱子分组显示

## 📁 输出文件

所有图表将保存为JPG格式（300 DPI）：
- `token_length/[dataset]_token_length_comparison.jpg`
- `serialize_time/[dataset]_serialization_speed.jpg`
- `train_time/[dataset]_training_efficiency.jpg`

## ➕ 添加新数据集

1. 在相应文件夹下添加新的CSV文件：
   - `token_length/[新数据集名]_token_length.csv`
   - `serialize_time/[新数据集名]_serialize_speed.csv`
   - `train_time/[新数据集名]_train_efficiency.csv`

2. 运行相应的画图脚本，会自动检测并处理新数据

## 🎨 图表样式

- 使用统一的matplotlib样式配置（来自`../plot_utils.py`）
- 字体：Arial 粗体
- 分辨率：300 DPI
- 格式：JPG
- **标注语言：英文**
- **横轴标签：不旋转**
- **序列化方法名称：基于src/algorithms/serializer/serializer_factory.py定义**
- **压缩比：范围2~15（1/压缩率）**
- **训练效率图：w/w/o BPE柱子分组显示**
- 统一的颜色方案和图例样式

## ⚠️ 注意事项

1. **中文字体警告**：运行时可能出现中文字体缺失警告，但不影响图表生成
2. **数据完整性**：确保CSV文件包含所有必需的列
3. **文件命名**：建议使用一致的命名规范（小写+下划线）

## 🔧 故障排除

### 常见问题
1. **ModuleNotFoundError**: 确保在正确的目录下运行脚本
2. **CSV文件未找到**: 检查CSV文件是否存在且命名正确
3. **图表不显示**: 使用`--show`参数或检查保存路径

### 调试模式
```bash
# 显示详细输出
python plot_token_length.py --verbose

# 仅测试数据加载
python -c "import pandas as pd; print(pd.read_csv('qm9hook_token_length.csv'))"
```

## 📈 实验结论模板

基于生成的图表，可以得出以下结论：

1. **序列长度**：BPE压缩比约为10倍（将序列长度压缩至原长度的约10%）
2. **序列化速度**：搜索和拓扑排序方法最快，cpp方法最慢
3. **训练效率**：序列化方法比传统图神经网络有约2倍加速比，w/w/o BPE方法分组对比显示

## 📞 联系方式

如有问题，请检查：
1. 数据格式是否正确
2. 依赖包是否安装完整
3. 文件路径是否正确
