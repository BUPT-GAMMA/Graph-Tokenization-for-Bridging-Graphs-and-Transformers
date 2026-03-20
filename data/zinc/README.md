# ZINC数据集转换文档

## 概述

本目录包含了从原始ZINC数据集转换而来的标准化分子图数据和SMILES字符串。转换过程将原始的DGL图格式转换为两种标准化的图格式，并生成四种不同形式的SMILES表示。

## 转换流程

```
原始ZINC数据集 (DGL格式)
    ↓ [dgl_graph_to_mol]
RDKit分子对象
    ↓ [mol_to_simplified_graph / mol_to_explicit_h_graph]
两种标准化图格式
    ↓ [generate_four_smiles_formats]
四种SMILES表示
```

## 输出文件说明

### 图数据集文件

#### 1. `simplified_graphs.pkl` - 简化图数据集
- **格式**: Pickle文件，包含(graph, label)元组列表
- **节点特征**: 原子序数 (6=C, 7=N, 8=O, 9=F, 15=P, 16=S, 17=Cl, 35=Br, 53=I)
- **边特征**: 化学键类型 (1=单键, 2=双键, 3=三键, 4=芳香键)
- **特点**: 
  - 使用原子序数作为节点特征，具有跨数据集通用性
  - 保持原始分子结构，不包含显式氢原子
  - 适合大多数分子图神经网络模型

#### 2. `explicit_h_graphs.pkl` - 显式氢原子图数据集
- **格式**: Pickle文件，包含(graph, label)元组列表
- **节点特征**: 原子序数 (包括氢原子=1)
- **边特征**: 化学键类型 (1=单键, 2=双键, 3=三键, 4=芳香键)
- **特点**:
  - 包含所有显式氢原子作为独立节点
  - 图规模更大，但信息更完整
  - 可以补偿原子序数特征的信息损失

### SMILES字符串文件

#### 1. `smiles_1_direct.txt` - 直接SMILES
- **格式**: 每行一个SMILES字符串
- **特点**: 标准SMILES格式，隐式氢原子
- **示例**: `COc1ccc2ccc(O)c(CN3CCN(S(=O)(=O)c4ccn(C)c4)CC3)c2c1`

#### 2. `smiles_2_explicit_h.txt` - 显式氢原子SMILES
- **格式**: 每行一个SMILES字符串
- **特点**: 显示所有氢原子，但不添加额外氢原子
- **示例**: `[CH3][O][c]1[cH][cH][c]2[cH][cH][c]([OH])[c]([CH2][N]3[CH2][CH2][N]([S](=[O])(=[O])[c]4[cH][cH][n]([CH3])[cH]4)[CH2][CH2]3)[c]2[cH]1`

#### 3. `smiles_3_addhs.txt` - AddHs SMILES
- **格式**: 每行一个SMILES字符串
- **特点**: 添加所有可能的氢原子后生成SMILES
- **示例**: `[H]Oc1c([H])c([H])c2c([H])c([H])c(OC([H])([H])[H])c([H])c2c1C([H])([H])N1C([H])([H])C([H])([H])N(S(=O)(=O)c2c([H])c([H])n(C([H])([H])[H])c2[H])C([H])([H])C1([H])[H]`

#### 4. `smiles_4_addhs_explicit_h.txt` - AddHs+显式氢原子SMILES
- **格式**: 每行一个SMILES字符串
- **特点**: 添加氢原子并显式标注所有氢原子
- **示例**: `[H][O][c]1[c]([H])[c]([H])[c]2[c]([H])[c]([H])[c]([O][C]([H])([H])[H])[c]([H])[c]2[c]1[C]([H])([H])[N]1[C]([H])([H])[C]([H])([H])[N]([S](=[O])(=[O])[c]2[c]([H])[c]([H])[n]([C]([H])([H])[H])[c]2[H])[C]([H])([H])[C]1([H])[H]`

## 数据统计

详细统计信息请查看 `conversion_stats.json` 文件。

## 使用示例

```python
import pickle
import torch

# 加载简化图数据
with open('simplified_graphs.pkl', 'rb') as f:
    simplified_graphs = pickle.load(f)

# 加载显式氢原子图数据
with open('explicit_h_graphs.pkl', 'rb') as f:
    explicit_h_graphs = pickle.load(f)

# 访问第一个分子
graph, label = simplified_graphs[0]
print(f"节点数: {graph.num_nodes()}")
print(f"边数: {graph.num_edges()}")
print(f"节点特征: {graph.ndata['feat']}")
print(f"边特征: {graph.edata['feat']}")
print(f"标签: {label}")

# 读取SMILES文件
with open('smiles_1_direct.txt', 'r') as f:
    smiles_list = f.read().strip().split('\n')
print(f"第一个分子SMILES: {smiles_list[0]}")
```

## 数据格式对比

| 特征 | 原始ZINC | 简化图 | 显式氢原子图 |
|------|----------|--------|-------------|
| 节点特征 | 28种原子类型 | 原子序数 | 原子序数(含H) |
| 边特征 | 4种键类型 | 4种键类型 | 4种键类型 |
| 氢原子处理 | 隐式 | 隐式 | 显式节点 |
| 图规模 | 中等 | 中等 | 较大 |
| 通用性 | 低 | 高 | 高 |

## 注意事项

1. **双向边**: 所有图都使用双向边表示，实际化学键数 = 边数/2
2. **原子序数**: 简化图使用标准原子序数，便于跨数据集使用
3. **氢原子**: 显式氢原子图包含所有氢原子，图规模显著增大
4. **SMILES对应**: 四种SMILES格式与图数据一一对应 