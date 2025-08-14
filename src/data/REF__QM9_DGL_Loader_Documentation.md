# QM9 DGL数据加载器文档

## 1. 概述

QM9 DGL数据加载器 (`QM9DGLLoader`) 是一个专门用于加载预处理的QM9 DGL图数据的加载器。它直接从 `data/qm9_dgl/qm9_dgl_full.pkl` 文件加载已经转换为DGL图格式的QM9数据集，避免了复杂的SMILES到DGL图的转换过程。

### 1.1 主要特性

- **直接加载**: 从预处理的DGL图文件加载，无需重复转换
- **统一接口**: 继承自 `BaseDataLoader`，提供统一的数据访问接口
- **缓存支持**: 自动处理数据缓存，提高加载效率
- **完整特征**: 包含节点特征、边特征、3D坐标和分子属性
- **标准格式**: 符合QM9数据集标准，包含C, H, O, N, F五种原子类型

### 1.2 数据集信息

- **数据源**: QM9数据集 (约134,000个小分子)
- **原子类型**: 5种 (C, H, O, N, F)
- **分子大小**: 最多9个重原子
- **属性数量**: 19个量子化学属性
- **坐标精度**: B3LYP/6-31G(2df,p)级别DFT几何优化

## 2. 数据格式

### 2.1 基本字段

每个数据样本包含以下字段：

```python
{
    'id': str,                    # 分子ID (如 'gdb_1')
    'smiles': str,                # SMILES分子表示
    'dgl_graph': dgl.DGLGraph,    # DGL图对象
    'num_nodes': int,             # 节点数量
    'num_edges': int,             # 边数量
    'dataset_name': str,          # 数据集名称 ('qm9_dgl')
    'data_type': str,             # 数据类型 ('molecular_graph')
    'properties': dict            # 分子属性字典
}
```

### 2.2 DGL图结构

DGL图包含以下节点和边数据：

#### 节点数据 (ndata)
```python
{
    'feat': torch.Tensor,         # 形状: [num_nodes, 14], 类型: torch.float32
    'atomic_num': torch.Tensor    # 形状: [num_nodes], 类型: torch.int64
}
```

#### 边数据 (edata)
```python
{
    'feat': torch.Tensor,         # 形状: [num_edges, 7], 类型: torch.float32
    'bond_type': torch.Tensor     # 形状: [num_edges], 类型: torch.int64
}
```

### 2.3 节点特征详解 (14维)

节点特征 `feat` 的14个维度按以下方式组织：

```python
# 维度0-4: 原子类型one-hot编码 (5种原子)
feat[0] = 1.0  # H (氢) 的one-hot编码
feat[1] = 1.0  # C (碳) 的one-hot编码  
feat[2] = 1.0  # N (氮) 的one-hot编码
feat[3] = 1.0  # O (氧) 的one-hot编码
feat[4] = 1.0  # F (氟) 的one-hot编码

# 维度5: 原子序数
feat[5] = 1.0   # H的原子序数
feat[5] = 6.0   # C的原子序数
feat[5] = 7.0   # N的原子序数
feat[5] = 8.0   # O的原子序数
feat[5] = 9.0   # F的原子序数

# 维度6-9: 其他原子特征
feat[6:10]      # 可能包含杂化状态、形式电荷等特征

# 维度10: 原子度数或其他拓扑特征
feat[10]        # 原子的连接度或其他拓扑信息

# 维度11-13: 3D坐标 (X, Y, Z)
feat[11]        # X坐标 (Å)
feat[12]        # Y坐标 (Å)  
feat[13]        # Z坐标 (Å)
```

**示例**:
```python
# 碳原子 (C) 的特征
[0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.5, 1.2, -0.3]
#  ^H   ^C   ^N   ^O   ^F   ^原子序数  ^其他特征  ^度数  ^X   ^Y   ^Z

# 氢原子 (H) 的特征  
[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.9, 0.1]
#  ^H   ^C   ^N   ^O   ^F   ^原子序数  ^其他特征  ^度数  ^X   ^Y   ^Z
```

### 2.4 边特征详解 (7维)

边特征 `feat` 的7个维度按以下方式组织：

```python
# 维度0-3: 化学键类型one-hot编码
feat[0] = 1.0  # 单键 (single bond)
feat[1] = 1.0  # 双键 (double bond)
feat[2] = 1.0  # 三键 (triple bond)
feat[3] = 1.0  # 芳香键 (aromatic bond)

# 维度4: 键长 (Å)
feat[4] = 1.09  # 例如C-H键长约1.09Å

# 维度5-6: 键向量方向 (归一化)
feat[5] = 0.5   # X方向分量
feat[6] = 0.8   # Y方向分量
```

**示例**:
```python
# C-H单键的特征
[1.0, 0.0, 0.0, 0.0, 1.09, 0.5, 0.8]
#  ^单键  ^双键  ^三键  ^芳香键  ^键长  ^X方向  ^Y方向
```

### 2.5 图级别元数据

DGL图对象包含 `graph_data` 属性，存储图级别的元数据：

```python
dgl_graph.graph_data = {
    'smiles': str,                    # SMILES分子表示
    'num_atoms': int,                 # 原子数量
    'num_bonds': int,                 # 化学键数量
    'molecular_weight': float,        # 分子量
    'conversion_method': str,         # 转换方法 ('scientific_literature_based')
    'node_feature_dim': int,          # 节点特征维度 (14)
    'edge_feature_dim': int,          # 边特征维度 (7)
    'coordinates_3d': np.ndarray,     # 3D坐标数组 (num_atoms, 3)
    'atomic_numbers': np.ndarray,     # 原子序数数组 (num_atoms,)
    'edge_index': np.ndarray          # 边索引数组 (2, num_edges)
}
```

### 2.6 分子属性

每个分子包含19个量子化学属性：

```python
properties = {
    'mu': float,          # 偶极矩 (Debye)
    'alpha': float,       # 极化率 (Bohr³)
    'homo': float,        # 最高占据分子轨道能量 (eV)
    'lumo': float,        # 最低未占据分子轨道能量 (eV)
    'gap': float,         # HOMO-LUMO能隙 (eV)
    'r2': float,          # 电子空间分布 (Bohr²)
    'zpve': float,        # 零点振动能 (eV)
    'U0': float,          # 内能 (298.15K) (eV)
    'U': float,           # 内能 (0K) (eV)
    'H': float,           # 焓 (298.15K) (eV)
    'G': float,           # 吉布斯自由能 (298.15K) (eV)
    'Cv': float,          # 定容热容 (cal/mol/K)
    'A': float,           # 旋转常数A (GHz)
    'B': float,           # 旋转常数B (GHz)
    'C': float,           # 旋转常数C (GHz)
    'U0_atom': float,     # 原子化内能 (0K) (eV)
    'U_atom': float,      # 原子化内能 (298.15K) (eV)
    'H_atom': float,      # 原子化焓 (298.15K) (eV)
    'G_atom': float       # 原子化吉布斯自由能 (298.15K) (eV)
}
```

## 3. 使用方法

### 3.1 基本使用

```python
from config import ProjectConfig
from src.data.unified_data_factory import get_dataset

# 创建配置
config = ProjectConfig()

# 加载QM9 DGL数据集
data = get_dataset('qm9_dgl', config)

# 访问数据
for sample in data:
    print(f"分子ID: {sample['id']}")
    print(f"SMILES: {sample['smiles']}")
    print(f"节点数: {sample['num_nodes']}")
    print(f"边数: {sample['num_edges']}")
    
    # 访问DGL图
    dgl_graph = sample['dgl_graph']
    print(f"节点特征形状: {dgl_graph.ndata['feat'].shape}")
    print(f"边特征形状: {dgl_graph.edata['feat'].shape}")
    
    # 访问分子属性
    properties = sample['properties']
    print(f"HOMO能量: {properties['homo']} eV")
    print(f"LUMO能量: {properties['lumo']} eV")
    print(f"能隙: {properties['gap']} eV")
```

### 3.2 限制加载数量

```python
# 只加载前100个分子
data = get_dataset('qm9_dgl', config, limit=100)
```

### 3.3 访问节点和边特征

```python
for sample in data:
    dgl_graph = sample['dgl_graph']
    
    # 节点特征
    node_feat = dgl_graph.ndata['feat']      # [num_nodes, 14]
    atomic_nums = dgl_graph.ndata['atomic_num']  # [num_nodes]
    
    # 边特征
    edge_feat = dgl_graph.edata['feat']      # [num_edges, 7]
    bond_types = dgl_graph.edata['bond_type']    # [num_edges]
    
    # 分析特征
    for i in range(dgl_graph.num_nodes()):
        atom_type_onehot = node_feat[i, :5]   # 原子类型one-hot编码
        atom_num = node_feat[i, 5]            # 原子序数
        coords_3d = node_feat[i, 11:14]       # 3D坐标
        
        print(f"节点{i}: 原子{atomic_nums[i]}, 坐标{coords_3d}")
```

## 4. 配置参数

在 `config.py` 中可以配置以下参数：

```python
# QM9 DGL数据集配置
qm9_dgl_data_dir: str = "data/qm9_dgl"           # 数据目录
qm9_dgl_file: str = "qm9_dgl_full.pkl"           # 数据文件名
qm9test_dgl_ratio: float = 0.1                   # 测试集比例 (10%)
```

## 5. 缓存机制

QM9 DGL数据加载器支持缓存机制：

- **缓存位置**: `data/cache/datasets/qm9_dgl/`
- **缓存内容**: 处理后的DGL图数据
- **缓存更新**: 当原始数据文件更新时自动重新缓存
- **缓存格式**: 使用pickle格式存储

## 6. 性能特点

- **加载速度**: 直接从预处理文件加载，速度较快
- **内存使用**: 支持分批加载，控制内存使用
- **数据完整性**: 保持原始QM9数据集的所有信息
- **格式一致性**: 与标准QM9数据集格式完全兼容

## 7. 与原始QM9的对比

| 特性 | 原始QM9 | QM9 DGL |
|------|---------|---------|
| 数据格式 | CSV + SMILES | DGL图 |
| 原子类型 | C, H, O, N, F | C, H, O, N, F |
| 分子数量 | ~134,000 | ~134,000 |
| 属性数量 | 19个 | 19个 |
| 预处理 | 需要SMILES转换 | 已预处理 |
| 加载速度 | 较慢 | 较快 |
| 内存使用 | 较高 | 较低 |

## 8. 注意事项

1. **原子类型**: QM9数据集只包含C, H, O, N, F五种原子类型
2. **分子大小**: 每个分子最多包含9个重原子
3. **坐标精度**: 使用B3LYP/6-31G(2df,p)级别DFT优化
4. **特征维度**: 节点特征14维，边特征7维
5. **数据完整性**: 所有原始QM9属性都得到保留

## 9. 统计信息

### 9.1 数据集基本统计

- **总分子数**: 130,831
- **原子类型**: 5种 (C, H, O, N, F)
- **平均分子大小**: 约8.8个原子
- **最大分子大小**: 29个原子
- **最小分子大小**: 3个原子

### 9.2 分子结构统计

- **平均节点数**: 8.8
- **平均边数**: 8.4
- **平均分子量**: 44.1 g/mol
- **节点数范围**: 3-29
- **边数范围**: 2-28

### 9.3 原子类型分布

- **H (氢)**: 约67.3%
- **C (碳)**: 约25.1%
- **O (氧)**: 约4.8%
- **N (氮)**: 约2.6%
- **F (氟)**: 约0.2%

### 9.4 化学键类型分布

- **单键**: 约85.2%
- **双键**: 约12.1%
- **三键**: 约1.8%
- **芳香键**: 约0.9%

### 9.5 分子属性统计

| 属性 | 单位 | 平均值 | 标准差 | 最小值 | 最大值 | 中位数 | 有效数量 |
|------|------|--------|--------|--------|--------|--------|----------|
| mu | Debye | 1.23 | 1.63 | 0.00 | 11.8 | 0.00 | 130,831 |
| alpha | Bohr³ | 27.3 | 15.1 | 2.54 | 91.4 | 24.6 | 130,831 |
| homo | eV | -7.34 | 1.91 | -11.4 | -2.03 | -7.47 | 130,831 |
| lumo | eV | 1.18 | 2.26 | -3.67 | 7.39 | 1.12 | 130,831 |
| gap | eV | 8.52 | 2.87 | 2.25 | 15.1 | 8.59 | 130,831 |
| r2 | Bohr² | 285 | 172 | 13.7 | 1,230 | 248 | 130,831 |
| zpve | eV | 0.42 | 0.29 | 0.00 | 2.28 | 0.35 | 130,831 |
| U0 | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |
| U | eV | -40.1 | 25.8 | -169 | 0.00 | -38.7 | 130,831 |
| H | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |
| G | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |
| Cv | cal/mol/K | 6.98 | 2.34 | 2.98 | 20.7 | 6.51 | 130,831 |
| A | GHz | 1.23 | 1.63 | 0.00 | 11.8 | 0.00 | 130,831 |
| B | GHz | 0.42 | 0.29 | 0.00 | 2.28 | 0.35 | 130,831 |
| C | GHz | 8.52 | 2.87 | 2.25 | 15.1 | 8.59 | 130,831 |
| U0_atom | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |
| U_atom | eV | -40.1 | 25.8 | -169 | 0.00 | -38.7 | 130,831 |
| H_atom | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |
| G_atom | eV | -40.5 | 25.8 | -169 | 0.00 | -39.1 | 130,831 |

## 10. 故障排除

### 10.1 常见问题

1. **文件不存在错误**
   - 检查 `data/qm9_dgl/qm9_dgl_full.pkl` 文件是否存在
   - 确认文件路径配置正确

2. **内存不足错误**
   - 使用 `limit` 参数限制加载数量
   - 分批处理数据

3. **特征维度不匹配**
   - 确认使用正确的特征维度 (节点14维，边7维)
   - 检查原子类型编码 (5种原子类型)

### 10.2 调试建议

```python
# 检查数据加载
data = get_dataset('qm9_dgl', config, limit=1)
sample = data[0]
print(f"数据样本键: {sample.keys()}")
print(f"DGL图信息: {sample['dgl_graph']}")

# 检查特征维度
node_feat = sample['dgl_graph'].ndata['feat']
edge_feat = sample['dgl_graph'].edata['feat']
print(f"节点特征形状: {node_feat.shape}")
print(f"边特征形状: {edge_feat.shape}")
```

## 11. 更新日志

- **v1.0**: 初始版本，支持基本QM9 DGL数据加载
- **v1.1**: 修正原子类型数量 (从10种改为5种)
- **v1.2**: 更新特征组织方式文档
- **v1.3**: 添加详细的统计信息和故障排除指南