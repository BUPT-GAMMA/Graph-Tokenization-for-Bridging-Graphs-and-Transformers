# 图数据导出与加载格式规范
## 严格约定，不做任何兼容性检查

---

## 📁 文件命名约定

### 导出文件命名
```
<dataset_name>_export.pkl
```

**示例：**
- `qm9_export.pkl`
- `zinc_export.pkl` 
- `molhiv_export.pkl`

---

## 📊 导出数据格式规范

### 文件结构
```python
{
    'graphs': List[Dict],     # 图数据列表
    'labels': List[Any],      # 标签列表  
    'splits': Dict[str, np.ndarray]  # 数据划分
}
```

### 图数据格式 (`graphs`)
每个图的格式：
```python
{
    'src': np.ndarray,        # 形状: (E,), 类型: int64, 源节点ID
    'dst': np.ndarray,        # 形状: (E,), 类型: int64, 目标节点ID
    'num_nodes': int,         # 节点总数
    'node_feat': np.ndarray,  # 形状: (N, D_node), 类型: int64 - 节点token特征
    'edge_feat': np.ndarray,  # 形状: (E, D_edge), 类型: int64 - 边token特征
}
```

**严格要求：**
- `src`, `dst`: 必须是 `np.int64` 类型的1D数组，长度相等
- `num_nodes`: 必须是 Python `int`
- `node_feat`: 必须是 `np.int64` 类型的2D数组，形状 `(N, D_node)`
  - `D_node` ≥ 1，大多数数据集 `D_node = 1`，部分数据集（如MNIST）`D_node > 1`
- `edge_feat`: 必须是 `np.int64` 类型的2D数组，形状 `(E, D_edge)`
  - `D_edge` ≥ 1，大多数数据集 `D_edge = 1`
- `N` = `num_nodes`, `E` = `len(src)` = `len(dst)`

### 标签格式 (`labels`)
**原则：保持原本格式导出，各数据集加载函数负责标签处理**

#### 单值回归数据集 (ZINC, AQSOL)
```python
labels: List[float]  # 每个元素是 Python float
```

#### 单值分类数据集 (MOLHIV, COLORS3, PROTEINS)
```python
labels: List[int]    # 每个元素是 Python int
```

#### 多属性回归数据集 (QM9)
```python
labels: List[Dict[str, float]]  # 每个元素是属性字典
# 示例: [{'mu': 0.123, 'alpha': 4.567}, ...]
```

#### 多任务回归数据集 (LRGB Peptides-struct)
```python
labels: List[List[float]]  # 每个元素是回归值列表
# 示例: [[0.12, 0.34, 0.56], [0.78, 0.90, 0.11], ...]
```

#### 多目标分类数据集 (LRGB Peptides-func)
```python
labels: List[List[int]]  # 每个元素是类别列表
# 示例: [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], ...]
```

### 数据划分格式 (`splits`)
```python
{
    'train': np.ndarray,  # 形状: (train_size,), 类型: int64, 训练集索引
    'val': np.ndarray,    # 形状: (val_size,), 类型: int64, 验证集索引  
    'test': np.ndarray,   # 形状: (test_size,), 类型: int64, 测试集索引
}
```

---

## 🔄 转换输出格式规范

### DGL格式 (`to_dgl_<dataset>()`)

**函数命名约定：**
```python
def to_dgl_qm9(data: Dict[str, Any]) -> List[Tuple[dgl.DGLGraph, Any]]
def to_dgl_zinc(data: Dict[str, Any]) -> List[Tuple[dgl.DGLGraph, Any]]
def to_dgl_molhiv(data: Dict[str, Any]) -> List[Tuple[dgl.DGLGraph, Any]]
# 每个数据集一个函数
```

**DGL图规范：**
- **图结构**: `dgl.graph((src_tensor, dst_tensor), num_nodes=N)`
- **节点特征**: `graph.ndata['feat']` → `torch.Tensor`, 形状: `(N, D_node)`, 类型: `int64`
- **边特征**: `graph.edata['feat']` → `torch.Tensor`, 形状: `(E, D_edge)`, 类型: `int64`
- **标签**: 由各数据集函数处理后返回适当格式

注意，如果这部分有公共函数可以提取，那么应该使优先调用公共函数，然后再在各个数据集的转换中加入他们自己的逻辑。

### PyG格式 (`to_pyg_<dataset>()`)

**函数命名约定：**
```python
def to_pyg_qm9(data: Dict[str, Any]) -> List[torch_geometric.data.Data]
def to_pyg_zinc(data: Dict[str, Any]) -> List[torch_geometric.data.Data] 
def to_pyg_molhiv(data: Dict[str, Any]) -> List[torch_geometric.data.Data]
# 每个数据集一个函数
```

**PyG Data对象规范：**
- **边索引**: `data.edge_index` → `torch.Tensor`, 形状: `(2, E)`, 类型: `int64`
- **节点特征**: `data.x` → `torch.Tensor`, 形状: `(N, D_node)`, 类型: `int64`
- **边特征**: `data.edge_attr` → `torch.Tensor`, 形状: `(E, D_edge)`, 类型: `int64`
- **节点数**: `data.num_nodes` → `int`
- **标签**: `data.y` → `torch.Tensor`（由各数据集函数决定具体格式）

类似的。如果有公共的函数可以提取，那么也是一样。

并且注意，因为原本的数据数据集格式是dgl图，那么如果要转成pyg，我们可能需要仔细谨慎的考虑清楚pg要怎么用。

---

## ⚡ 实现原则

### 严格断言
```python
# 导出时断言
assert isinstance(src, np.ndarray) and src.dtype == np.int64
assert isinstance(node_feat, np.ndarray) and node_feat.dtype == np.int64
assert node_feat.shape[0] == num_nodes and node_feat.ndim == 2
assert isinstance(edge_feat, np.ndarray) and edge_feat.dtype == np.int64
assert len(src) > 0, "图必须有边"
assert edge_feat.shape[0] == len(src) and edge_feat.ndim == 2

# 加载时断言  
assert isinstance(graph['src'], np.ndarray), "src必须是numpy数组"
assert graph['src'].dtype == np.int64, "src必须是int64类型"
assert len(graph['src']) > 0, "图必须有边"
assert isinstance(graph['node_feat'], np.ndarray), "node_feat必须是numpy数组"
assert graph['node_feat'].dtype == np.int64, "node_feat必须是int64类型"
assert graph['node_feat'].shape[0] == graph['num_nodes'], "节点特征数量错误"
```

### 零容错原则
- **不做类型转换**: 格式不对直接报错
- **不做形状调整**: 维度不对直接报错  
- **不做兼容处理**: 字段缺失直接报错
- **不做默认值**: 数据为空直接报错

### 高效转换
- 直接使用 `torch.from_numpy()` 进行零拷贝转换
- 避免Python循环，使用numpy/torch向量化操作
- 预分配内存，避免动态扩容

---

## 📝 数据集特定规范

### QM9
- **文件名**: `qm9_export.pkl`
- **节点特征**: 原子序数 (1-118), 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 键类型 (1-4: SINGLE, DOUBLE, TRIPLE, AROMATIC), 形状: `(E, 1)`, 类型: `int64`
- **标签**: 16个分子属性的字典 `Dict[str, float]`

### ZINC  
- **文件名**: `zinc_export.pkl`
- **节点特征**: 原子序数, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 键类型, 形状: `(E, 1)`, 类型: `int64`
- **标签**: logP_SA_cycle_normalized 回归值 `float`

### MOLHIV
- **文件名**: `molhiv_export.pkl`  
- **节点特征**: 原子序数, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 键类型, 形状: `(E, 1)`, 类型: `int64`
- **标签**: HIV抑制活性 `int` (0 或 1)

### MNIST
- **文件名**: `mnist_export.pkl`
- **节点特征**: [pixel_id, y_coord, x_coord], 形状: `(N, 3)`, 类型: `int64`
- **边特征**: 距离值, 形状: `(E, 1)`, 类型: `int64`  
- **标签**: 数字类别 `int` (0-9)

### Peptides-func (LRGB)
- **文件名**: `peptides_func_export.pkl`
- **节点特征**: 节点token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 10维多目标分类 `List[int]`

### Peptides-struct (LRGB)  
- **文件名**: `peptides_struct_export.pkl`
- **节点特征**: 节点token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 多任务回归 `List[float]`

### AQSOL
- **文件名**: `aqsol_export.pkl`
- **节点特征**: 原子序数, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 键类型 (0-4: NONE, SINGLE, DOUBLE, TRIPLE, AROMATIC), 形状: `(E, 1)`, 类型: `int64`
- **标签**: 溶解度回归值 `float`

### COLORS3
- **文件名**: `colors3_export.pkl`  
- **节点特征**: 颜色值, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 边类型, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 图分类 `int` (0-10, 11类)

### PROTEINS
- **文件名**: `proteins_export.pkl`
- **节点特征**: 蛋白质节点token, 形状: `(N, 1)`, 类型: `int64`  
- **边特征**: 边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 蛋白质功能分类 `int` (0 或 1)

### DD (Protein Structure)
- **文件名**: `dd_export.pkl`
- **节点特征**: 蛋白质节点token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 边token, 形状: `(E, 1)`, 类型: `int64` 
- **标签**: 酶功能分类 `int` (0 或 1)

### Mutagenicity
- **文件名**: `mutagenicity_export.pkl`
- **节点特征**: 原子token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 键token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 致突变性分类 `int` (0 或 1)

### CODE2 (OGB)
- **文件名**: `code2_export.pkl`
- **节点特征**: 代码AST节点双通道token, 形状: `(N, 2)`, 类型: `int64`
- **边特征**: 语法边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 序列预测任务 `List[int]` 或 `Any`

### COIL-DEL  
- **文件名**: `coildel_export.pkl`
- **节点特征**: 视觉特征双通道token, 形状: `(N, 2)`, 类型: `int64`
- **边特征**: 边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 物体分类 `int` (0-99, 100类)

### DBLP
- **文件名**: `dblp_export.pkl`
- **节点特征**: 学术网络节点token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 关系边token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 二分类 `int` (0 或 1)

### Twitter
- **文件名**: `twitter_export.pkl`
- **节点特征**: 社交网络节点token, 形状: `(N, 1)`, 类型: `int64`  
- **边特征**: 社交关系token, 形状: `(E, 1)`, 类型: `int64`
- **标签**: 社交网络分类 `int` (0 或 1)

### SYNTHETIC
- **文件名**: `synthetic_export.pkl`
- **节点特征**: 合成图节点token, 形状: `(N, 1)`, 类型: `int64`
- **边特征**: 合成图边token, 形状: `(E, 1)`, 类型: `int64`  
- **标签**: 合成图分类 `int` (0 或 1)

---

## 🧪 验证脚本

每个导出脚本必须包含格式验证：
```python
def validate_export_format(data):
    """验证导出数据格式"""
    # 验证顶层结构
    assert 'graphs' in data
    assert 'labels' in data  
    assert 'splits' in data
    
    # 验证图数据
    for i, graph in enumerate(data['graphs']):
        assert isinstance(graph['src'], np.ndarray), f"图{i} src格式错误"
        assert graph['src'].dtype == np.int64, f"图{i} src类型错误"
        assert isinstance(graph['dst'], np.ndarray), f"图{i} dst格式错误" 
        assert graph['dst'].dtype == np.int64, f"图{i} dst类型错误"
        assert len(graph['src']) == len(graph['dst']), f"图{i} 边数量不匹配"
        
        assert len(graph['src']) > 0, f"图{i} 必须有边"
        
        assert isinstance(graph['node_feat'], np.ndarray), f"图{i} node_feat格式错误"
        assert graph['node_feat'].dtype == np.int64, f"图{i} node_feat类型错误"  
        assert graph['node_feat'].shape[0] == graph['num_nodes'], f"图{i} 节点特征数量错误"
        
        assert isinstance(graph['edge_feat'], np.ndarray), f"图{i} edge_feat格式错误"
        assert graph['edge_feat'].dtype == np.int64, f"图{i} edge_feat类型错误"
        assert graph['edge_feat'].shape[0] == len(graph['src']), f"图{i} 边特征数量错误"
    
    # 验证数据划分
    for split_name in ['train', 'val', 'test']:
        assert split_name in data['splits'], f"缺少{split_name}划分"
        assert isinstance(data['splits'][split_name], np.ndarray), f"{split_name}划分格式错误"
        assert data['splits'][split_name].dtype == np.int64, f"{split_name}划分类型错误"
        
    print("✅ 格式验证通过")
```

---

## 🚀 使用流程

1. **导出**: `python export_system/export_<dataset>.py`
2. **验证**: 自动调用验证函数
啊？3. **使用**: `from export_system.loader import load_data, to_dgl_<dataset>, to_pyg_<dataset>`

**示例：**
```python
from export_system.loader import load_data, to_dgl_qm9, to_pyg_zinc

# 加载数据
qm9_data = load_data('qm9_export.pkl')
zinc_data = load_data('zinc_export.pkl')

# 转换为DGL格式
dgl_graphs = to_dgl_qm9(qm9_data)

# 转换为PyG格式  
pyg_data = to_pyg_zinc(zinc_data)
```

**零配置，零假设，零兼容，严格约定！**
