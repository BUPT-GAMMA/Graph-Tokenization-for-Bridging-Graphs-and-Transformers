# 图数据导出与加载系统

这是一个严格的图数据导出与加载系统，完全保持原有pipeline的数据不变，仅将节点/边token统一为`feat`特征。

## 🎯 核心原则

- **零改动原则**: 不改动图数据、标签数据、划分数据
- **唯一改动**: 仅将节点/边token作为`feat`特征保存
- **严格校验**: 确保导出数据与原始data loader结果完全一致
- **零容错**: 格式不符合要求时直接报错

## 📁 文件结构

```
export_system/
├── FORMAT_SPECIFICATION.md       # 格式规范文档
├── README.md                      # 本文件
├── base_exporter.py              # 基础导出器类
├── true_exporter.py              # 真正正确的导出器实现
├── loader.py                     # 统一加载器和格式转换
├── validate_format.py            # 格式验证器
├── unified_export_and_validate.py # 统一导出和校验脚本
└── __init__.py                   # 模块接口
```

## 🚀 使用方法

### 导出和校验所有数据集

```bash
# 运行统一脚本，导出所有数据集并进行校验
python export_system/unified_export_and_validate.py
```

该脚本会：
1. 依次导出所有16个数据集
2. 对每个导出结果进行严格校验
3. 确保导出数据与原始data loader结果完全一致
4. 输出详细的统计报告

### 加载和使用导出的数据

```python
from export_system import load_data, to_dgl_qm9, to_pyg_zinc

# 加载数据
qm9_data = load_data('data/exported/qm9_export.pkl')
zinc_data = load_data('data/exported/zinc_export.pkl')

# 转换为DGL格式
dgl_graphs = to_dgl_qm9(qm9_data)

# 转换为PyG格式  
pyg_data = to_pyg_zinc(zinc_data)

# 访问原始数据划分
train_indices = qm9_data['splits']['train']
val_indices = qm9_data['splits']['val']
test_indices = qm9_data['splits']['test']
```

### 单独验证文件

```bash
# 验证单个导出文件
python export_system/validate_format.py data/exported/qm9_export.pkl --strict
```

## 📊 支持的数据集

系统支持16个数据集，完全使用各自原有的data loader：

### 分子图数据集
- **QM9**: 分子属性预测 (16个属性的多标签回归)
- **ZINC**: 分子属性预测 (单值回归)  
- **MOLHIV**: HIV抑制活性预测 (二分类)
- **AQSOL**: 溶解度预测 (回归)
- **Mutagenicity**: 致突变性预测 (二分类)

### 生物图数据集
- **PROTEINS**: 蛋白质功能预测 (二分类)
- **DD**: 酶功能预测 (二分类)
- **Peptides-func**: 多功能预测 (多标签分类)
- **Peptides-struct**: 结构预测 (多任务回归)

### 其他图数据集
- **COLORS3**: 颜色图分类 (11分类)
- **CODE2**: 代码图序列预测
- **COIL-DEL**: 物体识别 (100分类)
- **DBLP**: 学术网络分析 (二分类)
- **Twitter**: 社交网络分析 (二分类)
- **MNIST**: 图像图分类 (10分类)
- **SYNTHETIC**: 合成图分类 (二分类)

## 🔧 数据格式

### 导出格式
每个导出文件包含以下结构，**完全保持原始data loader的格式**：

```python
{
    'graphs': List[Dict],           # 图数据列表（保持原结构）
    'labels': List[Any],            # 标签列表（保持原格式）
    'splits': Dict[str, np.ndarray] # 数据划分（保持原划分）
}
```

### 图数据格式
每个图包含以下字段，**仅feat字段是统一的token特征**：

```python
{
    'src': np.ndarray,        # 形状: (E,), 类型: int64 - 保持原样
    'dst': np.ndarray,        # 形状: (E,), 类型: int64 - 保持原样 
    'num_nodes': int,         # 节点总数 - 保持原样
    'node_feat': np.ndarray,  # 形状: (N, D_node), 类型: int64 - 节点token特征
    'edge_feat': np.ndarray,  # 形状: (E, D_edge), 类型: int64 - 边token特征
}
```

## ⚡ 校验机制

系统包含严格的校验机制，确保导出数据的正确性：

### 格式校验
- 验证数据结构完整性
- 验证数组类型和形状
- 验证索引范围和一致性

### 内容校验  
- **图结构**: 节点数、边数、连接关系完全一致
- **标签数据**: 与原始loader的标签完全相同
- **数据划分**: 严格保持原始train/val/test划分
- **特征维度**: token特征维度正确

### 一致性校验
- 随机采样验证图结构一致性
- 标签格式适配比较
- 数值精度校验

## ⚠️ 重要约定

### 严格原则
- **不改动任何原始数据**: 图结构、标签、划分完全保持不变
- **唯一改动**: 使用data loader标准接口获取预计算token，保存为统一的`feat`特征
- **零容错**: 任何不一致都会导致校验失败
- **原生兼容**: 完全兼容原有数据pipeline的使用方式

### Token特征约定
- **节点token**: `(N, D_node)`形状的`np.int64`数组，通常`D_node=1`
- **边token**: `(E, D_edge)`形状的`np.int64`数组，通常`D_edge=1`
- **多维token**: 少数数据集（如CODE2, COIL-DEL）`D_node`或`D_edge`大于1
- **类型**: 严格要求`np.int64`类型
- **边数约束**: 所有图必须至少有1条边

## 🐛 故障排除

### 常见问题

1. **导出失败**: 检查data loader是否正常工作，配置是否正确
2. **校验失败**: 通常说明token计算逻辑有误，检查特征提取
3. **格式错误**: 确保所有数组都是int64类型和正确形状
4. **划分不一致**: 确保完全使用data loader的原始划分

### 调试建议

1. 查看详细日志定位问题  
2. 使用`--debug`标志获取更多信息
3. 检查具体数据集的data loader实现
4. 验证token计算逻辑的正确性

## 📄 技术架构

```
原始Data Loader → TrueExporter → 导出文件 → 校验 → 确认一致性
      ↓              ↓            ↓        ↓         ↓
   保持原样     仅统一token    格式验证   内容校验   使用无忧
```

系统设计完全遵循"最小改动，最大兼容"的原则，确保与现有pipeline的完美集成。