# BPE动态压缩使用指南

## 概述

新版本实现了**动态BPE压缩**功能，将BPE压缩从预处理阶段移到了运行时的dataset transform中。这样的设计带来了以下优势：

- ✅ **统一数据集** - 只需要一个序列化数据集，BPE压缩在运行时动态应用
- ✅ **灵活配置** - 可以通过命令行参数灵活控制BPE压缩行为
- ✅ **随机性支持** - 支持训练时随机压缩，评估时确定性压缩
- ✅ **超参数搜索** - BPE参数可以作为超参数进行搜索优化

---

## 架构变化

### 之前的设计
```
图数据 → 序列化 → 原始序列数据集
       ↘          ↘
        BPE压缩 → BPE压缩数据集

两个独立的数据集，在数据准备阶段静态生成
```

### 现在的设计
```
图数据 → 序列化 → 序列化数据集
                   ↓
                运行时 → BPE Transform (可选)
                   ↓
                 训练/推理

动态压缩，通过配置控制
```

---

## 基本使用

### 1. 无BPE压缩（传统方式）

```bash
# 预训练
python run_pretrain.py --dataset qm9test --method feuler

# 微调
python run_finetune.py --dataset qm9test --method feuler --task regression
```

### 2. 启用BPE压缩（基本配置）

```bash
# 预训练
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000

# 微调（需要与预训练保持一致）
python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_num_merges 2000
```

---

## BPE参数详解

### 基础参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--bpe_num_merges` | BPE合并次数，0表示不使用BPE | 2000 | 整数值 |
| `--bpe_encode_backend` | BPE编码后端 | cpp | python, cpp |

### 编码策略参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--bpe_encode_rank_mode` | 排序模式 | all | all, topk, random, gaussian |
| `--bpe_encode_rank_k` | Top-K参数 | None | 整数 |
| `--bpe_encode_rank_min` | 随机范围最小值 | None | 整数 |
| `--bpe_encode_rank_max` | 随机范围最大值 | None | 整数 |
| `--bpe_encode_rank_dist` | 随机分布类型 | None | 字符串 |

### 评估模式参数

| 参数 | 说明 | 默认值 | 选项 |
|------|------|--------|------|
| `--bpe_eval_mode` | 评估时编码模式 | None | all, topk |
| `--bpe_eval_topk` | 评估时Top-K参数 | None | 整数 |

---

## 使用场景示例

### 1. 确定性压缩（用于基准测试）

```bash
# 预训练和微调都使用完全确定性的压缩
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000 \
  --bpe_encode_rank_mode all

python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_num_merges 2000 \
  --bpe_encode_rank_mode all
```

### 2. Top-K压缩（平衡性能和多样性）

```bash
# 只使用最频繁的1000个merge rules
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000 \
  --bpe_encode_rank_mode topk --bpe_encode_rank_k 1000

python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_num_merges 2000 \
  --bpe_encode_rank_mode topk --bpe_encode_rank_k 1000
```

### 3. 随机压缩（增强数据多样性）

```bash
# 训练时使用随机压缩，评估时使用确定性压缩
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000 \
  --bpe_encode_rank_mode random --bpe_encode_rank_min 100 --bpe_encode_rank_max 2000

python run_finetune.py --dataset qm9test --method feuler --task regression --bpe_num_merges 2000 \
  --bpe_encode_rank_mode random --bpe_encode_rank_min 100 --bpe_encode_rank_max 2000 \
  --bpe_eval_mode all  # 评估时使用确定性编码
```

### 4. 高斯分布随机压缩

```bash
# 使用高斯分布进行随机采样
python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000 \
  --bpe_encode_rank_mode gaussian --bpe_encode_rank_min 500 --bpe_encode_rank_max 1500
```

---

## JSON配置示例

### 基本BPE配置

```json
{
  "serialization": {
    "bpe": {
      "enabled": true,
      "encode_backend": "cpp",
      "encode_rank_mode": "all"
    }
  }
}
```

### Top-K配置

```json
{
  "serialization": {
    "bpe": {
      "enabled": true,
      "encode_backend": "cpp", 
      "encode_rank_mode": "topk",
      "encode_rank_k": 1000
    }
  }
}
```

### 随机压缩配置

```json
{
  "serialization": {
    "bpe": {
      "enabled": true,
      "encode_backend": "cpp",
      "encode_rank_mode": "random",
      "encode_rank_min": 100,
      "encode_rank_max": 2000,
      "eval_mode": "all"
    }
  }
}
```

### 使用JSON配置

```bash
# 保存配置到文件
cat > bpe_config.json << EOF
{
  "serialization": {
    "bpe": {
      "enabled": true,
      "encode_rank_mode": "topk",
      "encode_rank_k": 1000
    }
  },
  "bert": {
    "pretraining": {
      "epochs": 15,
      "learning_rate": 2e-4
    }
  }
}
EOF

# 使用配置文件
python run_pretrain.py --dataset qm9test --method feuler --config_json bpe_config.json
```

---

## 并行脚本使用

### 修改batch_pretrain_simple.py

```python
# 配置BPE压缩（合并次数>0表示使用BPE）
BPE_NUM_MERGES = 2000  # BPE合并次数
BPE_CONFIG = {
    "bpe_encode_rank_mode": "topk",
    "bpe_encode_rank_k": 1000,
}
```

### 修改batch_finetune_simple.py

```python
# 保持与预训练一致
BPE_NUM_MERGES = 2000  # 与预训练相同的合并次数
BPE_CONFIG = {
    "bpe_encode_rank_mode": "topk",  # 训练时配置
    "bpe_eval_mode": "all",          # 评估时确定性
    "bpe_encode_rank_k": 1000,
}
```

---

## 超参数搜索

BPE参数已经集成到超参数搜索中：

```python
# 在hyperparam_search.py中修改搜索空间
PRETRAIN_SEARCH_SPACE = {
    "method": ["feuler", "eulerian"],
    "epochs": [5, 8],
    "batch_size": [32, 64],
    "learning_rate": [1e-4, 2e-4],
    # BPE超参数
    "bpe_num_merges": [0, 1000, 2000],  # 0表示不使用BPE
    "bpe_encode_rank_mode": ["all", "topk"],
    "bpe_encode_rank_k": [500, 1000, 2000],
}
```

运行搜索：

```bash
python hyperparam_search.py
```

---

## 最佳实践

### 1. 训练策略

- **预训练**: 可以使用随机压缩增加数据多样性
- **微调**: 建议使用确定性压缩保证结果可重现
- **评估**: 始终使用确定性压缩

### 2. 参数选择

- **encode_rank_mode**:
  - `all`: 最完整的压缩，确定性
  - `topk`: 平衡压缩率和多样性
  - `random`: 最大多样性，适合预训练
  - `gaussian`: 更平滑的随机分布

- **encode_rank_k**: 
  - 小值(500-1000): 快速压缩，可能损失信息
  - 大值(1500-2000): 更完整压缩，计算量大

### 3. 一致性要求

- **预训练和微调必须使用相同的BPE设置**
- 如果预训练使用BPE，微调也必须启用BPE
- 词表大小和merge rules必须一致

### 4. 性能优化

- **C++后端**: 比Python后端快5-10倍
- **批量编码**: 在DataLoader中使用worker并行
- **缓存策略**: BPE Transform会自动缓存编码结果

---

## 故障排查

### 常见错误

1. **BPE codebook不存在**
   ```
   ❌ BPE Transform创建失败: BPE codebook不存在
   💡 请确保已构建BPE codebook
   ```
   **解决**: 运行 `python data_prepare.py` 构建BPE codebook

2. **预训练和微调BPE设置不一致**
   ```
   ❌ 词表大小不匹配
   ```
   **解决**: 确保微调时的BPE设置与预训练完全一致

3. **参数冲突**
   ```
   ❌ encode_rank_k只在encode_rank_mode=topk时有效
   ```
   **解决**: 检查参数组合的合理性

### 调试技巧

1. **查看BPE配置**:
   ```bash
   python run_pretrain.py --dataset qm9test --method feuler --show_config
   ```

2. **测试BPE功能**:
   ```bash
   # 先用小数据集测试
   python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 1000 --epochs 1
   ```

3. **比较有无BPE的结果**:
   ```bash
   # 无BPE
   python run_pretrain.py --dataset qm9test --method feuler --experiment_name "no_bpe"
   
   # 有BPE  
   python run_pretrain.py --dataset qm9test --method feuler --bpe_num_merges 2000 --experiment_name "with_bpe"
   ```

---

## 性能建议

### 内存优化

- BPE Transform会增加内存使用，建议适当减小batch_size
- 使用C++后端可以显著减少内存占用

### 计算优化

- 随机模式比确定性模式稍慢，但增加了多样性
- Top-K模式在k值较小时性能最好

### 存储优化

- 新架构下不再需要存储BPE压缩后的数据集
- 只需要存储BPE codebook（通常很小）

---

## 总结

动态BPE压缩系统提供了：

✅ **灵活性** - 运行时配置，无需重新处理数据  
✅ **可扩展性** - 支持多种压缩策略和超参数搜索  
✅ **一致性** - 统一的配置管理和参数传递  
✅ **性能** - C++后端和优化的并行处理  

这个新架构为BPE压缩的研究和应用提供了更强大和灵活的基础。

