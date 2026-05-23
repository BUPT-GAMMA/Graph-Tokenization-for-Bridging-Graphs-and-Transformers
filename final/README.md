# 实验数据加载器使用指南

## 概述

`experiment_data_loader.py` 是一个简单的实验数据加载器，用于从log目录中提取实验结果和配置信息，生成统一的DataFrame格式供分析使用。

## 主要功能

1. **自动扫描**: 扫描log目录结构，找到所有实验的metrics JSON文件
2. **智能解析**: 从JSON文件中提取配置参数和性能指标
3. **统一格式**: 生成包含所有实验信息的DataFrame
4. **灵活过滤**: 支持按实验组、数据集、方法、前缀进行过滤

## 数据结构

### 输入：log目录结构
```
log/
├── {experiment_group}/
│   └── {experiment_name}/
│       └── {dataset}/
│           └── {method}/
│               └── {prefix}finetune/
│                   └── finetune_metrics.json
```

### 输出：DataFrame列

#### 基本实验信息
- `experiment_group`: 实验组名称
- `experiment_name`: 实验名称
- `dataset`: 数据集名称
- `method`: 序列化方法
- `bpe_method`: BPE方法 (从config中提取)
- `prefix`: 实验前缀
- `task_type`: 任务类型 (regression/classification等)

#### 训练信息
- `epochs`: 训练轮数
- `steps_per_epoch`: 每轮步数
- `total_train_time_sec`: 总训练时间
- `avg_epoch_time_sec`: 平均每轮时间

#### 验证信息
- `val_loss`: 验证损失
- `val_best_mae`: 最佳验证MAE

#### 测试指标 (不同聚合模式)
- `test_mae_avg`: 测试MAE (avg聚合)
- `test_mae_best`: 测试MAE (best聚合)
- `test_mae_learned`: 测试MAE (learned聚合)
- `test_accuracy_avg`: 测试准确率 (avg聚合)
- `test_roc_auc_avg`: 测试AUC (avg聚合)
- `test_ap_avg`: 测试AP (avg聚合)

#### 配置信息 (从config字段提取)
- `d_model`: 模型维度
- `n_heads`: 注意力头数
- `n_layers`: 层数
- `finetune_learning_rate`: 微调学习率
- `finetune_batch_size`: 微调批大小
- `encoder_type`: 编码器类型
- `bpe_num_merges`: BPE合并次数
- `target_property`: 目标属性

#### 文件路径
- `metrics_file`: metrics JSON文件路径
- `model_dir`: 模型保存目录

## 使用方法

### 基本使用

```python
from final.experiment_data_loader import load_experiment_data

# 加载所有实验
df = load_experiment_data()

# 加载特定实验组
df = load_experiment_data(experiment_groups=['main_comparison'])

# 加载特定数据集和方法
df = load_experiment_data(
    datasets=['qm9', 'zinc'],
    methods=['feuler', 'cpp']
)
```

### 高级过滤

```python
# 过滤GT实验 (feuler + cpp方法，all BPE)
gt_experiments = df[
    (df['method'].isin(['feuler', 'cpp'])) &
    (df['bpe_method'] == 'all')
]

# 过滤回归任务实验
regression_experiments = df[df['task_type'] == 'regression']

# 过滤特定数据集
qm9_experiments = df[df['dataset'] == 'qm9']
```

### 性能分析

```python
# 查看不同方法的MAE性能 (使用best聚合)
method_performance = df.groupby('method')['test_mae_best'].mean()
print(method_performance.sort_values())

# 查看不同数据集上的性能
dataset_performance = df.groupby('dataset')['test_mae_best'].mean()
print(dataset_performance.sort_values())

# 比较不同聚合模式的性能差异
agg_comparison = df[['test_mae_avg', 'test_mae_best', 'test_mae_learned']].mean()
print(agg_comparison)
```

## 实际使用示例

### 1. 生成主实验对比表格

```python
# 过滤出要对比的方法
main_methods = ['feuler', 'cpp', 'eulerian', 'fcpp']
main_df = df[df['method'].isin(main_methods)]

# 按数据集和方法聚合MAE性能
performance_table = main_df.pivot_table(
    values='test_mae_best',
    index='dataset',
    columns='method',
    aggfunc='mean'
)

print(performance_table)
```

### 2. 分析BPE方法效果

```python
# 比较不同BPE方法的效果
bpe_comparison = df.groupby(['method', 'bpe_method'])['test_mae_best'].mean()
bpe_comparison = bpe_comparison.unstack()
print(bpe_comparison)
```

### 3. 分析超参数影响

```python
# 查看学习率对性能的影响
lr_analysis = df.groupby('finetune_learning_rate')['test_mae_best'].mean()
print(lr_analysis.sort_values())

# 查看模型大小对性能的影响
size_analysis = df.groupby(['d_model', 'n_layers'])['test_mae_best'].mean()
print(size_analysis.sort_values())
```

## 注意事项

1. **数据类型**: 所有数值字段都是float类型，缺失值用NaN表示
2. **任务类型**: 根据`task_type`字段区分回归和分类任务
3. **聚合模式**: 测试指标包含三种聚合模式的结果
4. **配置完整性**: 不是所有实验都有完整的config信息，老实验可能缺失某些字段

## 扩展使用

可以配合pandas的强大功能进行更复杂的分析：

```python
# 时间序列分析
df['train_time_minutes'] = df['total_train_time_sec'] / 60
time_analysis = df.groupby('method')['train_time_minutes'].mean()

# 相关性分析
correlation_matrix = df[['d_model', 'test_mae_best', 'train_time_minutes']].corr()

# 统计显著性检验
from scipy import stats
method_a = df[df['method'] == 'feuler']['test_mae_best']
method_b = df[df['method'] == 'cpp']['test_mae_best']
t_stat, p_value = stats.ttest_ind(method_a, method_b)
```
