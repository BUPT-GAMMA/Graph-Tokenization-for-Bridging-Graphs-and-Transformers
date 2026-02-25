# LRGB数据集集成总结

## 📋 完成任务

本次成功集成了PyTorch Geometric的LRGB数据集中的两个数据集到TokenizerGraph项目：

### 数据集概览

| 数据集 | 图数量 | 节点数均值 | 边数均值 | 任务类型 | 标签维度 | 节点Token数 | 边Token数 |
|--------|--------|------------|----------|----------|----------|-------------|-----------|
| molhiv | 41,127 | 25.51 | 54.94 | 二分类 | 1 | 55 | 4 |
| peptides_func | 15,535 | 150.94 | 307.30 | 10维多标签分类 | 10 | 53 | 5 |
| peptides_struct | 15,535 | 150.94 | 307.30 | 11维多目标回归 | 11 | 53 | 5 |

### 关键技术成果

1. **存储优化**：从17GB压缩到15-16MB，压缩比>1000倍
2. **Token化高效性**：LRGB数据集特征高度离散，只需很少的token
3. **动态特征扫描**：自动发现和分配特征组合到token的映射

## 🔧 实现文件

### 预处理脚本
- `prepare_lrgb_data.py` - 主预处理脚本
- 实现动态特征扫描和token分配
- 轻量级存储格式，保存为`data.pkl.gz`

### 数据加载器
- `src/data/loader/peptides_func_loader.py` - Peptides-func加载器
- `src/data/loader/peptides_struct_loader.py` - Peptides-struct加载器
- 已在`src/data/unified_data_factory.py`中注册

### 分析工具
- `analyze_lrgb_datasets.py` - 统计分析脚本
- `analyze_lrgb_features.py` - 详细特征分析脚本
- `analyze_full_lrgb_features.py` - 完整数据集特征分析

## 📊 数据格式定义

### 存储格式
- **文件**: `data/<dataset>/data.pkl.gz` (gzip压缩的pickle)
- **结构**: `List[Tuple[lightweight_graph_dict, label_array]]`
- **索引**: `train_index.json`, `val_index.json`, `test_index.json`
- **Token映射**: `token_mappings.json`

### 标签格式
- **Peptides-func**: shape=(10,)的numpy数组 → 转换为10维列表（多标签分类）
- **Peptides-struct**: shape=(11,)的numpy数组 → 转换为11维列表（多目标回归）

### 图数据格式
轻量级图数据字典包含：
```python
{
    'num_nodes': int,
    'num_edges': int, 
    'edges': (src_array, dst_array),  # numpy int32
    'node_features': node_feat_array,  # numpy float32
    'edge_features': edge_feat_array,  # numpy float32
    'node_token_ids': node_tokens,     # numpy int32, shape=(N,1)
    'edge_token_ids': edge_tokens,     # numpy int32, shape=(E,1)
    'node_type_ids': node_types,       # numpy int32, shape=(N,)
    'edge_type_ids': edge_types,       # numpy int32, shape=(E,)
}
```

## 🎯 关键经验

### 编程原则
1. **严格数据契约** - 预处理定义什么，加载器就期望什么
2. **fail-fast原则** - 数据格式不对直接报错，不做兼容性处理
3. **避免假设性编程** - 不写大量if-else处理不确定情况
4. **单一数据源** - 预处理阶段定义唯一的数据格式

详细编程原则见 `CODING_STANDARDS.md` 第5章。

### 技术发现
1. **LRGB特征高度离散** - 234万个节点只有53种特征组合
2. **Token化策略简单** - 基于特征组合哈希的直接映射
3. **存储优化巨大** - numpy+gzip比DGL图对象小1000倍

## 📁 生成的分析报告

- `LRGB_DATASETS_STATS.md` - 数据集统计对比表
- `LRGB_CLASS_BALANCE_REPORT.md` - 分类数据集类别平衡分析
- `lrgb_analysis_results.json` - 详细分析结果JSON

## 🔄 使用示例

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

cfg = ProjectConfig()

# 加载数据
udi = UnifiedDataInterface(cfg, "peptides_func")
train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = \
    udi.get_sequences_by_splits(method="feuler")

# 获取词汇表
vocab_manager = udi.get_vocab(method="feuler")
```

---

*本集成基于PyTorch Geometric 2.4.0版本的LRGBDataset，参考文档：*  
*https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.datasets.LRGBDataset.html*
