# 数据存储结构回退总结

## 决策背景

经过深入分析，我们认识到原计划的数据存储结构改动（引入 `splits/<version>/seed{seed}.json` 层级）价值有限：

- 增加了不必要的目录层级复杂度
- 需要迁移现有数据
- 多种子支持在实际中很少用到
- 原有的简单结构已经足够清晰且够用

## 保持原有结构

```
data/<dataset>/
├── data.pkl           # 完整数据集（图结构 + 属性）
├── train_index.json   # 训练集索引
├── val_index.json     # 验证集索引
└── test_index.json    # 测试集索引
```

## 已完成的改动

### ✅ 保留的改进
1. **UnifiedDataInterface** - 提供统一的数据访问接口
2. **职责分离** - UDI 负责对外接口，DataLoader 负责底层加载
3. **移除 quick_interface 依赖** - BERT 训练流程直接使用 UDI
4. **错误信息修正** - BaseDataLoader 使用通用错误文案

### ❌ 回退的改动
1. 删除 `src/data/splits.py`
2. 删除 `scripts/generate_splits.py`
3. 不引入新的目录层级或文件格式

## UnifiedDataInterface 适配

内部方法 `_load_split_indices()` 直接读取原有的三个索引文件：

```python
def _load_split_indices(self) -> Dict[str, List[int]]:
    """加载原有格式的划分索引文件"""
    data_dir = Path("data") / self.dataset
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        index_path = data_dir / f"{split_name}_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                splits[split_name] = json.load(f)
        else:
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
    
    return splits
```

## 优势

- **零迁移成本** - 完全向后兼容
- **结构简单** - 易于理解和维护
- **专注核心价值** - 接口统一而非存储结构改变
- **测试通过** - 所有相关测试已更新并通过

## 总结

这是一个务实的决策：保留真正有价值的改进（统一接口），避免过度设计（文件结构改动）。


