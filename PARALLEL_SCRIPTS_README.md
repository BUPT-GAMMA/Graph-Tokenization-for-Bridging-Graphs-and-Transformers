# 并行脚本使用指南

## 概述

为了快速打通流程并支持多GPU并行训练，提供了以下简单的并行脚本：

- **`batch_pretrain_simple.py`** - 多GPU并行预训练
- **`batch_finetune_simple.py`** - 多GPU并行微调  
- **`hyperparam_search.py`** - 超参数搜索

这些脚本的特点：
- ✅ **简单直接** - 配置在脚本顶部，易于修改
- ✅ **GPU并行** - 每个GPU运行1-2个任务
- ✅ **统一实验组** - 便于结果管理
- ✅ **实时监控** - 显示任务进度和结果
- ✅ **信号处理** - 支持Ctrl+C安全中断

---

## 1. 并行预训练 (`batch_pretrain_simple.py`)

### 基本使用

```bash
python batch_pretrain_simple.py
```

### 配置参数

在脚本顶部修改以下配置：

```python
# ===== 配置区域 =====
EXPERIMENT_GROUP = "parallel_pretrain"  # 实验组名称
DATASET = "qm9test"                     # 数据集
METHODS = ["feuler", "eulerian", "dfs", "bfs"]  # 要训练的方法
GPUS = [0, 1]                          # 可用的GPU编号

# 超参数配置（可选）
HYPERPARAMS = [
    {"epochs": 5, "batch_size": 32, "learning_rate": 1e-4},
    {"epochs": 8, "batch_size": 64, "learning_rate": 2e-4},
    # 可以添加更多超参数组合
]

# JSON配置文件（可选，优先级高于HYPERPARAMS）
JSON_CONFIG = None  # 或者 "pretrain_config.json"
```

### 工作原理

1. **任务生成**: 根据`METHODS`和`HYPERPARAMS`生成任务列表
2. **GPU分配**: 自动将任务分配到空闲的GPU上
3. **并行执行**: 每个GPU运行1个任务，完成后自动接收新任务
4. **结果监控**: 实时显示每个GPU的任务状态

### 示例输出

```
🚀 开始并行预训练...
实验组: parallel_pretrain
数据集: qm9test
可用GPU: [0, 1]
总任务数: 8

🚀 GPU 0: 开始任务 feuler_e5_b32_lr0.0001
🚀 GPU 1: 开始任务 eulerian_e5_b32_lr0.0001
[GPU0-feuler_e5_b32_lr0.0001] 🔧 初始化配置...
[GPU1-eulerian_e5_b32_lr0.0001] 🔧 初始化配置...
...
✅ GPU 0: 任务 feuler_e5_b32_lr0.0001 完成
🚀 GPU 0: 开始任务 dfs_e5_b32_lr0.0001
...
```

---

## 2. 并行微调 (`batch_finetune_simple.py`)

### 基本使用

```bash
python batch_finetune_simple.py
```

### 配置参数

```python
# ===== 配置区域 =====
EXPERIMENT_GROUP = "parallel_finetune"
DATASET = "qm9test"
METHODS = ["feuler", "eulerian", "dfs", "bfs"]
GPUS = [0, 1]

# 任务配置
TASK_TYPE = "regression"  # 或 "classification"
TARGET_PROPERTY = "homo"  # 回归任务的目标属性

# 微调超参数配置
FINETUNE_HYPERPARAMS = [
    {"finetune_epochs": 15, "finetune_batch_size": 16, "finetune_learning_rate": 2e-5},
    {"finetune_epochs": 20, "finetune_batch_size": 32, "finetune_learning_rate": 1e-5},
]
```

### 预训练模型检查

脚本会自动检查对应的预训练模型是否存在，如果缺少会提示先运行预训练。

---

## 3. 超参数搜索 (`hyperparam_search.py`)

### 基本使用

```bash
python hyperparam_search.py
```

### 配置参数

```python
# ===== 配置区域 =====
EXPERIMENT_GROUP = "hyperparam_search"
DATASET = "qm9test"
GPUS = [0, 1]

# 搜索任务类型
SEARCH_TYPE = "pretrain"  # "pretrain" 或 "finetune"

# 预训练超参数搜索空间
PRETRAIN_SEARCH_SPACE = {
    "method": ["feuler", "eulerian"],
    "epochs": [5, 8],
    "batch_size": [32, 64],
    "learning_rate": [1e-4, 2e-4],
    "hidden_size": [512],
}

# 搜索策略
SEARCH_STRATEGY = "grid"  # "grid" 或 "random"
MAX_TRIALS = 20  # 仅用于随机搜索
```

### 搜索策略

#### 网格搜索 (Grid Search)
- 遍历所有参数组合
- 适合参数空间较小的情况
- 结果完整但计算量大

#### 随机搜索 (Random Search)  
- 随机采样参数组合
- 适合参数空间较大的情况
- 计算效率高，可能找到更好的结果

### 结果文件

搜索结果保存在`hyperparam_results/`目录：

```
hyperparam_results/
├── search_config.json      # 搜索配置
├── search_results.json     # 搜索结果汇总
├── {task_name}_gpu0.log    # 各任务详细日志
└── {task_name}_gpu1.log
```

### 结果分析

```json
{
  "summary": {
    "total_tasks": 8,
    "successful_tasks": 7,
    "failed_tasks": 1,
    "success_rate": 0.875,
    "best_result": {
      "experiment_name": "feuler_epochs8_batch_size64_learning_rate0.0002",
      "params": {...},
      "best_val_loss": 1.2345
    }
  },
  "all_results": [...]
}
```

---

## 4. 快速批量实验流程

### 完整的预训练+微调流程

```bash
# 1. 批量预训练
python batch_pretrain_simple.py

# 2. 等待完成后，批量微调
python batch_finetune_simple.py
```

### 超参数搜索流程

```bash
# 1. 预训练超参数搜索
# 修改 hyperparam_search.py 中的 SEARCH_TYPE = "pretrain"
python hyperparam_search.py

# 2. 分析结果，选择最佳预训练参数

# 3. 微调超参数搜索  
# 修改 SEARCH_TYPE = "finetune"
python hyperparam_search.py
```

---

## 5. 脚本自定义

### 修改GPU配置

```python
# 单GPU
GPUS = [0]

# 多GPU
GPUS = [0, 1, 2, 3]

# 指定特定GPU
GPUS = [1, 3]  # 只使用GPU 1和3
```

### 修改方法列表

```python
# 只测试部分方法
METHODS = ["feuler", "eulerian"]

# 测试所有方法
METHODS = ["feuler", "eulerian", "dfs", "bfs", "cpp", "fcpp"]

# 单个方法
METHODS = ["feuler"]
```

### 添加超参数

```python
# 预训练超参数
HYPERPARAMS = [
    {"epochs": 5, "batch_size": 32, "learning_rate": 1e-4, "hidden_size": 512},
    {"epochs": 10, "batch_size": 64, "learning_rate": 2e-4, "hidden_size": 768},
    {"epochs": 15, "batch_size": 32, "learning_rate": 5e-5, "hidden_size": 512},
]

# 微调超参数
FINETUNE_HYPERPARAMS = [
    {"finetune_epochs": 10, "finetune_batch_size": 16, "finetune_learning_rate": 1e-5},
    {"finetune_epochs": 20, "finetune_batch_size": 32, "finetune_learning_rate": 2e-5},
    {"finetune_epochs": 30, "finetune_batch_size": 16, "finetune_learning_rate": 5e-6},
]
```

### 使用JSON配置

创建配置文件 `custom_config.json`:

```json
{
  "bert": {
    "architecture": {
      "hidden_size": 768,
      "num_hidden_layers": 8
    },
    "pretraining": {
      "epochs": 20,
      "learning_rate": 1e-4,
      "warmup_steps": 1000
    }
  },
  "system": {
    "mixed_precision": true
  }
}
```

然后在脚本中设置:

```python
JSON_CONFIG = "custom_config.json"
```

---

## 6. 监控和调试

### 实时监控

所有脚本都提供实时输出，显示：
- 任务启动信息
- GPU分配状态  
- 任务执行进度
- 完成/失败统计

### 日志文件

超参数搜索会为每个任务生成独立的日志文件，便于调试。

### 中断处理

使用`Ctrl+C`可以安全中断所有任务：
- 自动终止所有子进程
- 保存已完成的结果
- 清理临时文件

### 错误排查

1. **GPU内存不足**: 减小`batch_size`
2. **预训练模型不存在**: 先运行对应的预训练任务
3. **数据文件缺失**: 检查`data_prepare.py`是否已运行
4. **配置错误**: 检查参数组合是否合理

---

## 7. 扩展和定制

### 添加新的搜索参数

在`SEARCH_SPACE`中添加新参数：

```python
PRETRAIN_SEARCH_SPACE = {
    "method": ["feuler", "eulerian"],
    "epochs": [5, 10, 15],
    "batch_size": [16, 32, 64],
    "learning_rate": [1e-4, 2e-4, 5e-4],
    "hidden_size": [512, 768],
    "num_layers": [4, 6, 8],  # 新增参数
    "dropout": [0.1, 0.2],    # 新增参数
}
```

### 修改结果收集

在`monitor_process`函数中添加自定义的指标提取逻辑。

### 集成其他工具

这些脚本可以轻松集成到其他工具中：
- W&B超参数搜索
- Optuna优化
- 集群作业调度系统

---

## 总结

这些并行脚本提供了快速、简单的多GPU训练解决方案：

✅ **快速上手** - 修改配置即可运行  
✅ **并行高效** - 充分利用多GPU资源  
✅ **结果管理** - 统一的实验组和命名  
✅ **扩展灵活** - 易于定制和扩展  

适合快速原型验证、超参数搜索和批量实验等场景。


