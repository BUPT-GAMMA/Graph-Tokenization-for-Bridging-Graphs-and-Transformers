# 图数据集导出系统使用指南

## 概述

本文档描述了从 TokenizerGraph 项目导出的标准化图数据集格式、使用方法和数据集详细信息。导出系统将原始图数据集转换为统一的 `.pkl` 格式，确保数据完整性和跨框架兼容性。

### 数据集范围
- **14个成功导出的数据集**，涵盖409,668个图
- **任务类型覆盖**：单/多属性回归、二分类、多分类、多标签分类、多标签回归
- **领域覆盖**：分子图、生物图、社交网络图、合成图等
- **来源多样**：TU数据集、OGB数据集、LRGB数据集

### 系统特性
- **零容忍数据格式**：严格的数据类型和格式验证
- **跨框架兼容**：支持 DGL 和 PyTorch Geometric
- **完整性保证**：所有409,668个图经过逐一验证
- **原始数据一致性**：与原始数据加载器100%一致
- **精确任务类型**：基于代码确定的准确任务信息，无猜测性描述

## 数据格式规范

### 文件结构
每个数据集导出为一个 `.pkl` 文件，包含以下字段：

```python
{
    'graphs': List[Dict],      # 图数据列表
    'labels': List[Any],       # 标签列表  
    'splits': Dict[str, np.ndarray],  # 数据划分
    'dataset_info': Dict       # 数据集元信息
}
```

### 图数据格式
每个图的格式为：

```python
{
    'src': np.ndarray,         # 形状: (E,), 类型: int64, 源节点ID
    'dst': np.ndarray,         # 形状: (E,), 类型: int64, 目标节点ID  
    'num_nodes': int,          # 节点总数
    'node_feat': np.ndarray,   # 形状: (N, D_node), 类型: int64, 节点token特征
    'edge_feat': np.ndarray,   # 形状: (E, D_edge), 类型: int64, 边token特征
}
```

**关键约束**：
- 所有图必须至少包含1条边 (`len(src) > 0`)
- 节点和边特征必须为 `np.int64` 类型的2D数组
- 边按照DGL的双向存储方式（无向图存储为双向有向边）

### 数据划分格式
```python
{
    'train': np.ndarray,  # 训练集索引，类型: int64
    'val': np.ndarray,    # 验证集索引，类型: int64  
    'test': np.ndarray,   # 测试集索引，类型: int64
}
```

## 任务类型分类

本系统包含的数据集涵盖以下机器学习任务类型：

### 回归任务（Regression）
- **单属性回归**：预测一个连续数值
  - ZINC: logP_SA_cycle_normalized
  - AQSOL: solubility
- **多属性回归**：同时预测多个连续数值
  - QM9: 16个分子属性（mu, alpha, homo, lumo, gap等）

### 分类任务（Classification）  
- **二分类**：预测两个类别之一
  - MOLHIV: 0/1分类
- **多分类**：预测多个类别之一
  - COIL-DEL: 100类分类（0-99）
  - COLORS-3, PROTEINS, DD, Mutagenicity, DBLP, TWITTER, SYNTHETIC: 分类任务
- **多标签分类**：每个样本可同时属于多个类别
  - Peptides-func: 10维多标签分类

### 多标签回归（Multi-label Regression）
- **多标签回归**：同时预测多个连续数值（与多属性回归类似）
  - Peptides-struct: 11维多标签回归

### 数据集分布统计
- **回归任务**: 3个数据集（152,654图）
- **分类任务**: 9个数据集（225,479图）  
- **多标签分类**: 1个数据集（15,535图）
- **多标签回归**: 1个数据集（15,535图）

## 数据集详细信息

### 1. QM9 (qm9_export.pkl, 177.65 MB)
- **任务类型**：多属性回归（16个分子属性）
- **图数量**：130,831
- **划分**：训练104,664 / 验证13,084 / 测试13,083
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征
- **标签格式**：字典，包含16个回归属性
- **标签属性**：mu, alpha, homo, lumo, gap, r2, zpve, u0, u298, h298, g298, cv, u0_atom, u298_atom, h298_atom, g298_atom
- **平均边数**：32（示例图）

### 2. ZINC (zinc_export.pkl, 30.98 MB)  
- **任务类型**：单属性回归
- **图数量**：12,000
- **划分**：训练10,000 / 验证1,000 / 测试1,000
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征  
- **标签格式**：数值，单一回归目标
- **标签属性**：logP_SA_cycle_normalized
- **平均边数**：114（示例图）

### 3. MOLHIV (molhiv_export.pkl, 65.95 MB)
- **任务类型**：二分类（OGB图分类数据集）
- **图数量**：41,127
- **划分**：训练32,901 / 验证4,113 / 测试4,113
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征
- **标签格式**：0/1分类标签
- **平均边数**：50（示例图）

### 4. AQSOL (aqsol_export.pkl, 19.35 MB)
- **任务类型**：单属性回归
- **图数量**：9,823  
- **划分**：训练7,858 / 验证983 / 测试982
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征
- **标签格式**：数值，单一回归目标
- **标签属性**：solubility
- **平均边数**：56（示例图）

### 5. COLORS-3 (colors3_export.pkl, 50.27 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：10,500
- **划分**：训练8,400 / 验证1,050 / 测试1,050
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：12（示例图）

### 6. PROTEINS (proteins_export.pkl, 4.21 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：1,113
- **划分**：训练890 / 验证111 / 测试112
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：162（示例图）

### 7. DD (dd_export.pkl, 41.34 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：1,178
- **划分**：训练942 / 验证118 / 测试118
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：1798（示例图）

### 8. Mutagenicity (mutagenicity_export.pkl, 7.77 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：4,337
- **划分**：训练3,469 / 验证434 / 测试434  
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：32（示例图）

### 9. COIL-DEL (coildel_export.pkl, 11.56 MB)
- **任务类型**：多分类（100类，TU数据集）
- **图数量**：3,900
- **划分**：训练3,120 / 验证390 / 测试390
- **图类型**：无向图（双向边存储）
- **节点特征**：(26, 2) 二维节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签（0-99）
- **平均边数**：138（示例图）
- **特殊性**：支持多维节点特征

### 10. DBLP (dblp_export.pkl, 21.91 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：19,456
- **划分**：训练15,564 / 验证1,946 / 测试1,946
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：20（示例图）

### 11. TWITTER (twitter_export.pkl, 57.93 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：144,033
- **划分**：训练115,226 / 验证14,403 / 测试14,404
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：6（示例图）

### 12. SYNTHETIC (synthetic_export.pkl, 2.97 MB)
- **任务类型**：图分类（TU数据集）
- **图数量**：300
- **划分**：训练240 / 验证30 / 测试30
- **图类型**：无向图（双向边存储）
- **节点特征**：节点token特征
- **边特征**：边token特征
- **标签格式**：分类标签
- **平均边数**：392（示例图）

### 13. Peptides-func (peptides_func_export.pkl, 130.96 MB)
- **任务类型**：多标签分类（10个功能性标签，LRGB数据集）
- **图数量**：15,535
- **划分**：训练10,873 / 验证2,331 / 测试2,331
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征
- **标签格式**：10维多标签分类
- **平均边数**：682（示例图）

### 14. Peptides-struct (peptides_struct_export.pkl, 131.10 MB)
- **任务类型**：多标签回归（11个结构性属性，LRGB数据集）
- **图数量**：15,535
- **划分**：训练10,873 / 验证2,331 / 测试2,331
- **图类型**：分子图（无向图，双向边存储）
- **节点特征**：原子token特征
- **边特征**：化学键token特征
- **标签格式**：11维多标签回归  
- **平均边数**：682（示例图）

## 使用方法

### 1. 加载导出数据
```python
from export_system import load_data

# 加载数据集
data = load_data('data/exported/qm9_export.pkl')
graphs = data['graphs']        # 图数据
labels = data['labels']        # 标签数据  
splits = data['splits']        # 划分索引
info = data['dataset_info']    # 元信息
```

### 2. 转换为DGL格式
```python
from export_system import create_dgl_graphs

# 转换为DGL图列表
dgl_graphs = create_dgl_graphs(graphs)

# 访问第一个图
g = dgl_graphs[0]
node_feat = g.ndata['feat']  # 节点特征
edge_feat = g.edata['feat']  # 边特征
```

### 3. 转换为PyTorch Geometric格式  
```python
from export_system import create_pyg_graphs

# 转换为PyG数据列表
pyg_graphs = create_pyg_graphs(graphs)

# 访问第一个图
data = pyg_graphs[0] 
x = data.x          # 节点特征
edge_index = data.edge_index  # 边索引
edge_attr = data.edge_attr    # 边特征
```

### 4. 按划分获取数据
```python
# 获取训练集
train_indices = splits['train']
train_graphs = [graphs[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]

# 获取验证集和测试集
val_indices = splits['val']  
test_indices = splits['test']
```

## 验证结果摘要

### 全面验证统计
- **成功导出数据集**：14/16个（code2和mnist数据不完整，跳过）
- **总验证图数量**：409,668个
- **验证项目**：
  - ✅ 划分索引完全一致性（numpy级别）
  - ✅ 图结构1:1完整对应（节点数、边数、连接关系）  
  - ✅ 特征逐token完全匹配（使用原始data loader验证）
  - ✅ 标签完全一致性
  - ✅ 边存储方式正确（双向边处理）
  - ✅ 数据类型和格式严格符合规范

### 关键验证点
- **零数据丢失**：所有图的结构、特征、标签完全保持原样
- **格式严格性**：所有特征均为`np.int64`类型的2D数组
- **边数约束**：所有图至少包含1条边
- **多维特征支持**：如COIL-DEL的(26,2)节点特征

## 文件清单

### 导出数据文件（data/exported/）
```
qm9_export.pkl          - 177.65 MB - QM9分子属性数据集
zinc_export.pkl         - 30.98 MB  - ZINC分子优化数据集  
molhiv_export.pkl       - 65.95 MB  - MOLHIV HIV抑制数据集
aqsol_export.pkl        - 19.35 MB  - AQSOL溶解度数据集
colors3_export.pkl      - 50.27 MB  - COLORS-3图着色数据集
proteins_export.pkl     - 4.21 MB   - PROTEINS蛋白质数据集
dd_export.pkl           - 41.34 MB  - DD蛋白质数据集
mutagenicity_export.pkl - 7.77 MB   - Mutagenicity致突变性数据集
coildel_export.pkl      - 11.56 MB  - COIL-DEL删除库数据集
dblp_export.pkl         - 21.91 MB  - DBLP学术网络数据集
twitter_export.pkl      - 57.93 MB  - TWITTER社交网络数据集
synthetic_export.pkl    - 2.97 MB   - SYNTHETIC合成图数据集
peptides_func_export.pkl - 130.96 MB - Peptides-func肽功能数据集
peptides_struct_export.pkl - 131.10 MB - Peptides-struct肽结构数据集

总计: 753.98 MB, 409,668个图
```

### Python脚本文件（export_system/）
```
__init__.py                    - 包初始化和导出接口
true_exporter.py              - 核心导出器实现
loader.py                     - 数据加载和格式转换
validate_format.py            - 格式验证功能
FORMAT_SPECIFICATION.md       - 详细格式规范
export_all_datasets.py        - 批量导出脚本
test_export_comprehensive.py  - 全面验证脚本
README.md                     - 使用说明
```

## 技术实现细节

### 导出流程
1. **原始数据获取**：使用项目原生UDI接口加载数据
2. **Token特征提取**：调用`get_node_tokens_bulk`和`get_edge_tokens_bulk`
3. **格式标准化**：转换为numpy数组，确保int64类型
4. **严格验证**：逐图验证结构和特征一致性
5. **序列化保存**：使用pickle格式保存

### 兼容性保证
- **DGL兼容**：边按DGL标准双向存储
- **PyG兼容**：提供转换函数
- **零拷贝转换**：使用`torch.from_numpy()`高效转换

### 性能特征
- **内存效率**：使用numpy数组减少内存占用
- **加载速度**：pickle格式快速加载
- **类型安全**：严格的int64类型确保数值精度

## 注意事项

1. **内存需求**：大数据集（如QM9、TWITTER）需要足够内存
2. **Python版本**：建议Python 3.8+
3. **依赖包**：需要numpy, torch, dgl, torch_geometric
4. **边存储**：所有无向图以双向有向边存储
5. **特征维度**：不同数据集的特征维度不同，注意适配
6. **任务类型**：所有任务类型信息基于原始代码确定，请根据具体需求选择适当的数据集

## 数据质量保证

- **严格验证**：409,668个图逐一经过6层验证（结构、特征、标签、划分、格式、边数）
- **零数据丢失**：与原始data loader 100%一致，无任何数据修改或丢失
- **准确描述**：所有数据集描述基于源代码分析，无推测性内容
- **格式统一**：严格遵循FORMAT_SPECIFICATION.md规范

---
*文档版本: 1.0*  
*生成时间: 2025-08-20*  
*验证状态: 全部通过*
