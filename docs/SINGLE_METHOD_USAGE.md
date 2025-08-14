# 单个方法运行脚本使用指南

## 概述

新版本提供了两个单个方法运行的脚本，用于替代之前的批量处理脚本：

- **`run_pretrain.py`** - 单个方法BERT预训练
- **`run_finetune.py`** - 单个方法BERT微调

这些脚本具有以下特点：
- ✅ **灵活的参数配置** - 支持命令行参数和高级配置覆盖
- ✅ **智能默认设置** - 自动推断任务类型、目标属性等
- ✅ **清晰的输出信息** - 详细的进度提示和结果报告
- ✅ **错误诊断** - 友好的错误提示和解决建议

---

## 预训练脚本 (`run_pretrain.py`)

### 基本使用

```bash
# 最基本的使用
python run_pretrain.py --dataset qm9test --method feuler

# 使用BPE压缩
python run_pretrain.py --dataset qm9test --method feuler --bpe

# 指定GPU设备
python run_pretrain.py --dataset qm9test --method feuler --device cuda:0
```

### 自定义训练参数

```bash
# 调整训练轮数和批次大小
python run_pretrain.py --dataset qm9test --method feuler \
  --epochs 15 --batch_size 64

# 调整学习率和掩码概率
python run_pretrain.py --dataset qm9test --method feuler \
  --learning_rate 2e-4 --mask_prob 0.2

# 调整早停参数
python run_pretrain.py --dataset qm9test --method feuler \
  --early_stopping_patience 5
```

### 自定义模型架构

```bash
# 调整BERT模型大小
python run_pretrain.py --dataset qm9test --method feuler \
  --hidden_size 768 --num_layers 8 --num_heads 12

# 调整序列长度
python run_pretrain.py --dataset qm9test --method feuler \
  --max_seq_length 128
```

### 实验管理

```bash
# 指定实验分组和名称
python run_pretrain.py --dataset qm9test --method feuler \
  --experiment_group "ablation_study" \
  --experiment_name "feuler_large_model"

# 系统配置
python run_pretrain.py --dataset qm9test --method feuler \
  --device cuda:1 --workers 8
```

### 高级配置覆盖

```bash
# 使用 --config_override 参数覆盖任意配置项
python run_pretrain.py --dataset qm9test --method feuler \
  --config_override \
    bert.pretraining.warmup_steps=1000 \
    bert.architecture.dropout=0.2 \
    system.device=cuda:1
```

### 支持的参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--dataset` | 数据集名称 | `qm9test`, `qm9`, `zinc` |
| `--method` | 序列化方法 | `feuler`, `eulerian`, `dfs` |
| `--bpe` | 使用BPE压缩 | 无值标志 |
| `--epochs` | 训练轮数 | `10`, `20` |
| `--batch_size` | 批次大小 | `32`, `64` |
| `--learning_rate` | 学习率 | `1e-4`, `2e-4` |
| `--hidden_size` | 隐藏层大小 | `512`, `768` |
| `--num_layers` | 层数 | `4`, `6`, `8` |
| `--num_heads` | 注意力头数 | `8`, `12` |
| `--config_override` | 高级配置覆盖 | `key=value` 格式 |

---

## 微调脚本 (`run_finetune.py`)

### 基本使用

```bash
# 基本回归任务（自动推断目标属性）
python run_finetune.py --dataset qm9test --method feuler --task regression

# 指定回归目标属性
python run_finetune.py --dataset qm9test --method feuler \
  --task regression --target_property homo

# 分类任务（自动推断类别数）
python run_finetune.py --dataset mnist --method feuler --task classification
```

### 自定义微调参数

```bash
# 调整微调参数
python run_finetune.py --dataset qm9test --method feuler --task regression \
  --finetune_epochs 20 --finetune_batch_size 16 --finetune_learning_rate 2e-5

# 调整早停参数
python run_finetune.py --dataset qm9test --method feuler --task regression \
  --finetune_patience 8
```

### 数据处理配置

```bash
# 标签标准化和池化方法
python run_finetune.py --dataset qm9test --method feuler --task regression \
  --normalization standard --pooling_method mean

# 其他标准化选项
python run_finetune.py --dataset qm9test --method feuler --task regression \
  --normalization minmax  # 或 robust
```

### 分类任务配置

```bash
# 指定分类类别数
python run_finetune.py --dataset custom --method feuler --task classification \
  --num_classes 5

# MNIST分类示例
python run_finetune.py --dataset mnist --method feuler --task classification
```

### 高级配置覆盖

```bash
# 微调相关的高级配置
python run_finetune.py --dataset qm9test --method feuler --task regression \
  --config_override \
    bert.finetuning.warmup_ratio=0.1 \
    bert.finetuning.weight_decay=0.01 \
    system.device=cuda:1
```

### 支持的参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | 任务类型 | `regression`, `classification` |
| `--target_property` | 回归目标属性 | `homo`, `lumo`, `solubility` |
| `--num_classes` | 分类类别数 | `2`, `10` |
| `--finetune_epochs` | 微调轮数 | `10`, `20` |
| `--finetune_batch_size` | 微调批次大小 | `16`, `32` |
| `--finetune_learning_rate` | 微调学习率 | `1e-5`, `2e-5` |
| `--normalization` | 标签标准化方法 | `standard`, `minmax`, `robust` |
| `--pooling_method` | 池化方法 | `cls`, `mean`, `max` |

---

## 配置覆盖系统

### 覆盖语法

使用 `--config_override` 参数可以覆盖配置文件中的任意参数：

```bash
--config_override key1=value1 key2=value2 key3.nested.key=value3
```

### 支持的配置路径

以下是一些常用的配置路径示例：

#### BERT架构配置
```bash
bert.architecture.hidden_size=768
bert.architecture.num_hidden_layers=8
bert.architecture.num_attention_heads=12
bert.architecture.hidden_dropout_prob=0.1
bert.architecture.attention_probs_dropout_prob=0.1
```

#### 预训练配置
```bash
bert.pretraining.epochs=15
bert.pretraining.batch_size=64
bert.pretraining.learning_rate=2e-4
bert.pretraining.warmup_steps=1000
bert.pretraining.mask_prob=0.15
```

#### 微调配置
```bash
bert.finetuning.epochs=20
bert.finetuning.batch_size=16
bert.finetuning.learning_rate=2e-5
bert.finetuning.warmup_ratio=0.1
bert.finetuning.weight_decay=0.01
```

#### 系统配置
```bash
system.device=cuda:1
system.num_workers=8
system.mixed_precision=true
```

#### 任务配置
```bash
task.type=regression
task.target_property=homo
task.normalization=standard
```

### 类型自动转换

配置覆盖系统会自动进行类型转换：

- **整数**: `epochs=10` → `10`
- **浮点数**: `learning_rate=1e-4` → `0.0001`
- **布尔值**: `bpe=true` → `True` (支持 `true/false`, `1/0`, `yes/no`, `on/off`)
- **字符串**: `device=cuda:1` → `"cuda:1"`

---

## 工作流程示例

### 完整的预训练 + 微调流程

```bash
# 1. 预训练
python run_pretrain.py --dataset qm9test --method feuler --bpe \
  --epochs 10 --batch_size 32 \
  --experiment_group "qm9_experiments"

# 2. 微调
python run_finetune.py --dataset qm9test --method feuler --bpe \
  --task regression --target_property homo \
  --finetune_epochs 15 --finetune_batch_size 16 \
  --experiment_group "qm9_experiments"
```

### 批量实验（不同方法）

```bash
# 对不同序列化方法进行实验
for method in feuler eulerian dfs bfs; do
  echo "Training with method: $method"
  
  # 预训练
  python run_pretrain.py --dataset qm9test --method $method --bpe
  
  # 微调
  python run_finetune.py --dataset qm9test --method $method --bpe \
    --task regression --target_property homo
done
```

### 超参数搜索

```bash
# 不同学习率实验
for lr in 1e-4 2e-4 5e-4; do
  python run_pretrain.py --dataset qm9test --method feuler \
    --learning_rate $lr \
    --experiment_name "feuler_lr_${lr}"
done
```

---

## 智能特性

### 自动推断功能

1. **任务类型推断**
   - 根据数据集元信息自动判断回归/分类任务
   - QM9 → 回归，MNIST → 分类

2. **目标属性推断**
   - QM9数据集默认使用 `homo` 属性
   - 其他数据集使用元信息中的默认属性

3. **类别数推断**
   - 分类任务自动从数据集元信息获取类别数
   - MNIST → 10类

4. **实验名称生成**
   - 自动生成格式：`{dataset}-{method}-{bpe/raw}`
   - 例如：`qm9test-feuler-bpe`

### 前置检查

1. **预训练模型验证**（微调时）
   - 自动检查预训练模型是否存在
   - 提供清晰的错误提示和建议

2. **配置验证**
   - 启动前验证所有配置参数
   - 及时发现配置错误

3. **数据文件检查**
   - 验证序列化数据和词表文件是否存在
   - 提供数据准备的建议

---

## 输出和日志

### 控制台输出

脚本会提供详细的进度信息：

```
🔧 初始化配置...
✅ 设置配置: bert.pretraining.epochs = 15
📋 配置摘要
============================================================
数据集: qm9test
序列化方法: feuler
BPE压缩: True
...
🚀 开始BERT预训练...
📂 加载序列化数据: feuler
✅ 数据加载完成:
  训练集: 1000 个序列
  验证集: 200 个序列
  测试集: 200 个序列
📚 加载词表...
✅ 词表加载成功: 5000 个token
🎓 开始模型训练...
✅ 预训练完成!
📊 最优验证损失: 2.1234
```

### 文件输出

训练结果会保存到标准的目录结构中：

```
model/
├── {experiment_group}/
│   └── {experiment_name}/
│       └── {dataset}/
│           └── {method}-{bpe/raw}/
│               ├── best/          # 最优模型
│               └── final/         # 最终模型

logs/
├── {experiment_group}/
│   └── {experiment_name}/
│       └── {dataset}/
│           └── {method}-{bpe/raw}/
│               ├── training.log
│               └── tensorboard/
```

---

## 错误处理和诊断

### 常见错误和解决方案

1. **数据文件不存在**
   ```
   ❌ 词表加载失败: 词表不存在: /path/to/vocab.json
   💡 请确保已运行数据准备流程构建词表
   ```
   **解决**: 运行 `python data_prepare.py` 构建数据

2. **预训练模型不存在**
   ```
   ❌ 未找到预训练模型
      已检查路径: /path/to/best, /path/to/final, /path/to/compat
   💡 请先运行预训练:
   python run_pretrain.py --dataset qm9test --method feuler --bpe
   ```
   **解决**: 先运行对应的预训练命令

3. **配置验证失败**
   ```
   ❌ 配置验证失败: 隐藏层大小必须是注意力头数的倍数
   ```
   **解决**: 调整 `--hidden_size` 和 `--num_heads` 参数

4. **GPU内存不足**
   ```
   ❌ 预训练失败: CUDA out of memory
   ```
   **解决**: 减小 `--batch_size` 或使用 `--device cpu`

### 调试技巧

1. **使用小数据集测试**
   ```bash
   python run_pretrain.py --dataset qm9test --method feuler --epochs 1
   ```

2. **检查配置**
   ```bash
   python run_pretrain.py --dataset qm9test --method feuler --help
   ```

3. **强制使用CPU**
   ```bash
   python run_pretrain.py --dataset qm9test --method feuler --device cpu
   ```

---

## 迁移指南

### 从批量脚本迁移

原来的批量脚本：
```bash
python pretrain_all_methods.py --dataset qm9test --version latest --variants both
```

新的单个方法脚本：
```bash
# 需要为每个方法单独运行
python run_pretrain.py --dataset qm9test --method feuler
python run_pretrain.py --dataset qm9test --method feuler --bpe
python run_pretrain.py --dataset qm9test --method eulerian
python run_pretrain.py --dataset qm9test --method eulerian --bpe
```

### 脚本自动化

可以使用shell脚本实现批量运行：

```bash
#!/bin/bash
# batch_pretrain.sh

DATASET="qm9test"
METHODS=("feuler" "eulerian" "dfs" "bfs")

for method in "${METHODS[@]}"; do
  echo "Training method: $method"
  
  # RAW版本
  python run_pretrain.py --dataset $DATASET --method $method
  
  # BPE版本
  python run_pretrain.py --dataset $DATASET --method $method --bpe
done
```

---

## 总结

新的单个方法运行脚本提供了：

✅ **更灵活的配置** - 支持细粒度的参数调整  
✅ **更清晰的接口** - 明确的参数和智能默认值  
✅ **更好的用户体验** - 详细的提示和错误诊断  
✅ **更强的扩展性** - 支持高级配置覆盖  

虽然失去了批量处理的便利性，但通过shell脚本等工具可以轻松实现批量运行，同时获得了更大的灵活性和控制力。


