# 效率对比实验修改记录

## 🔄 已完成的修改

### 1. 序列化方法名称标准化 ✅
- **前**: DFS, BFS, Topological, SMILES, Eulerian, Graph_Seq, CPP
- **后**: dfs, bfs, topo, smiles, eulerian, feuler, cpp
- **来源**: 基于`src/algorithms/serializer/serializer_factory.py`中的定义
- **影响文件**: 所有CSV数据文件

### 2. 图表标注语言改为英文 ✅
- **轴标签**: 中文 → 英文
- **图表标题**: 中文 → 英文  
- **图例标签**: 中文 → 英文
- **示例**:
  - "序列化方法" → "Serialization Method"
  - "序列长度" → "Sequence Length"
  - "训练时间" → "Training Time per Epoch (seconds)"

### 3. 横轴标签不旋转 ✅
- **修改**: 去掉了`plt.xticks(rotation=45, ha='right')`
- **效果**: 横轴标签水平显示，提高可读性

### 4. 压缩率改为压缩比 ✅
- **计算公式**: 压缩比 = 1 / 压缩率
- **数值范围**: 2~15（而不是0~100%）
- **轴标签**: "压缩率 (%)" → "Compression Ratio"
- **数值标签**: "10.5%" → "9.5x"

### 5. 训练效率图柱子分组 ✅
- **设计**: w和w/o BPE的柱子贴在一起，形成分组
- **实现**: 
  - 添加`method_group`列到CSV
  - 重新设计柱子位置算法
  - 序列化方法分组 + GNN方法单独显示
- **视觉效果**: 更清晰地对比有无BPE的效果

## 📊 修改后的数据格式

### token_length CSV格式
```csv
serialization_method,original_length,bpe_compressed_length,compression_rate
dfs,145.2,15.3,0.105
bfs,142.8,14.8,0.104
topo,98.5,12.1,0.123
smiles,87.3,11.2,0.128
eulerian,158.7,14.9,0.094
feuler,162.4,15.1,0.093
cpp,184.6,16.8,0.091
```

### serialize_time CSV格式
```csv
serialization_method,graphs_per_second
dfs,1250.3
bfs,1180.7
topo,1890.2
smiles,856.4
eulerian,678.9
feuler,645.2
cpp,312.8
```

### train_time CSV格式
```csv
method_name,epoch_time_seconds,method_type,method_group
dfs,45.2,serialization_no_bpe,dfs
dfs_bpe,38.7,serialization_with_bpe,dfs
eulerian,48.3,serialization_no_bpe,eulerian
eulerian_bpe,41.1,serialization_with_bpe,eulerian
cpp,52.8,serialization_no_bpe,cpp
cpp_bpe,44.6,serialization_with_bpe,cpp
GCN,89.4,graph_neural_network,gnn
GraphGPS,95.7,graph_neural_network,gnn
GraphMamba,87.2,graph_neural_network,gnn
```

## 🎯 修改后的图表特征

### 1. 序列化长度对比图
- 英文标注
- 横轴不旋转
- 压缩比显示（2~15范围）
- 三个柱子：Original Length, BPE Compressed Length, Compression Ratio

### 2. 序列化速度对比图  
- 英文标注
- 横轴不旋转
- 单柱显示graphs/second

### 3. 训练效率对比图
- 英文标注
- 横轴不旋转
- 分组柱状图：w/w/o BPE方法贴在一起
- 序列化方法 vs 图神经网络方法对比
- 显示平均加速比

## 🔧 技术实现要点

### 压缩比计算
```python
compression_ratios = [1.0/ratio for ratio in df['compression_rate'].tolist()]
ax2.set_ylim(2, 15)  # 设置压缩比范围
```

### 训练效率分组
```python
# 序列化方法的x坐标
serial_x = np.arange(n_serialization) * group_gap

# GNN方法的x坐标  
gnn_start = serial_x[-1] + method_gap if n_serialization > 0 else 0
gnn_x = np.arange(n_gnn) + gnn_start

# 分组柱子
bars_no_bpe = ax.bar(serial_x - bar_width/2, no_bpe_times, bar_width, ...)
bars_with_bpe = ax.bar(serial_x + bar_width/2, with_bpe_times, bar_width, ...)
```

## ✅ 验证结果

所有修改已通过测试：
- ✅ 脚本正常运行
- ✅ 图表成功生成  
- ✅ 格式符合要求
- ✅ 数据显示正确
- ✅ 英文标注清晰
- ✅ 柱子分组合理

## 📁 影响的文件

### 修改的脚本文件
- `token_length/plot_token_length.py`
- `serialize_time/plot_serialize_speed.py`
- `train_time/plot_train_efficiency.py` (重写)

### 修改的数据文件
- `token_length/qm9hook_token_length.csv`
- `serialize_time/qm9hook_serialize_speed.csv`
- `train_time/qm9hook_train_efficiency.csv`

### 更新的文档文件
- `README.md`
- `USAGE.md`
- `CHANGES.md` (新增)

## 🚀 使用方式

```bash
# 一键生成所有图表
cd /home/gzy/py/tokenizerGraph/final/exp1_speed
python run_all_plots.py

# 显示图表
python run_all_plots.py --show
```

所有修改已完成，图表符合论文要求！🎉


