# 基于最佳预训练参数的微调优化搜索

本文档说明如何使用从 `large_batch.db` 提取的最佳预训练参数来进行微调超参数优化搜索。

## 📁 相关文件

```
hyperopt/
├── scripts/
│   ├── extract_best_params_for_finetuning.py  # 提取最佳预训练参数
│   ├── finetune_hyperopt_with_seeds.py       # 基于种子的微调搜索
│   └── analyze_optuna_results.py             # 分析搜索结果
├── journal/
│   ├── large_batch.db                        # 预训练搜索数据库
│   └── finetune_with_seeds.db               # 微调搜索数据库（将生成）
└── results/
    ├── best_pretrain_params_for_finetuning.json  # 提取的最佳参数
    └── finetuning_search_ranges.json            # 建议的搜索范围
```

## 🚀 使用流程

### 步骤1: 提取最佳预训练参数

```bash
cd hyperopt/scripts
python extract_best_params_for_finetuning.py \
    --journal ../journal/large_batch.db \
    --target_study methods_large_batch_pretrain_all \
    --output_dir ../results
```

**提取结果:**
- ✅ **Top3 overall结果**: 都是feuler方法，loss从0.4022到0.4031
- ✅ **eulerian最佳**: lr=1.66e-4, bs=128, loss=0.4781
- ✅ **feuler最佳**: lr=3.23e-4, bs=128, loss=0.4022 (整体最佳)
- ✅ **cpp最佳**: lr=3.60e-4, bs=256, loss=0.5838
- ✅ **fcpp最佳**: lr=3.15e-4, bs=256, loss=0.5700
- ⚠️  **默认配置**: 需要单独训练预训练模型

### 步骤2: 运行微调优化搜索

```bash
python finetune_hyperopt_with_seeds.py \
    --bpe_mode all \
    --journal_file ../journal/finetune_with_seeds.db \
    --seed_file ../results/best_pretrain_params_for_finetuning.json \
    --pretrain_journal ../journal/large_batch.db \
    --trials 50
```

## 🎯 种子参数设计

脚本会自动将预训练参数转换为微调参数：

| 参数 | 预训练 → 微调 | 说明 |
|------|---------------|------|
| Learning Rate | `lr × 0.1` | 微调学习率通常更小 |
| Batch Size | `min(bs, 128)` | 微调批次不要太大 |
| Weight Decay | `wd × 0.5` | 微调权重衰减适度减小 |
| Gradient Norm | `grad_norm` | 保持不变 |
| Warmup Ratio | `warmup_ratio × 0.5` | 微调预热比例较小 |
| Epochs | `5` | 默认微调轮数 |

## 🔧 微调搜索空间

| 参数 | 搜索范围 | 类型 |
|------|----------|------|
| Learning Rate | [1e-5, 1e-3] | log scale |
| Batch Size | [32, 64, 128] | categorical |
| Weight Decay | [0.0, 0.3] | uniform |
| Gradient Norm | [0.5, 3.0] | uniform |
| Warmup Ratio | [0.0, 0.3] | uniform |
| Epochs | [3, 5, 10, 15] | categorical |

## 📊 监控和分析

### 查看搜索进度
```bash
python analyze_optuna_results.py \
    --journal ../journal/finetune_with_seeds.db \
    --bpe_mode all
```

### 结果文件
- **微调搜索数据库**: `hyperopt/journal/finetune_with_seeds.db`
- **最佳参数JSON**: `hyperopt/results/best_pretrain_params_for_finetuning.json`
- **搜索范围建议**: `hyperopt/results/finetuning_search_ranges.json`

## 🎨 核心特性

### 智能种子系统
- 🌱 **自动种子转换**: 将预训练参数智能转换为微调参数
- 🔍 **模型路径查找**: 自动找到对应的预训练模型路径
- 🚫 **跳过无效种子**: 自动跳过没有预训练模型的种子

### 并发优化
- 🔄 **并发安全**: 支持多进程并行搜索
- 🎯 **TPE采样器**: 使用贝叶斯优化进行高效搜索
- ✂️ **智能剪枝**: 30%分位数剪枝器避免浪费资源

### 目标优化
- 📈 **测试集指标**: 优先使用MAE/RMSE作为优化目标
- ⏱️ **时间记录**: 记录每次试验的训练时间
- 📝 **详细日志**: 完整的搜索过程日志

## ❗ 注意事项

1. **预训练模型依赖**: 种子试验需要对应的预训练模型存在
2. **序列化方法**: 确保微调时使用与预训练时相同的序列化方法  
3. **资源管理**: 微调搜索可能需要大量GPU时间，建议合理设置trials数量
4. **默认配置**: 如需使用默认配置种子，需要先用默认参数训练预训练模型

## 🔗 相关脚本

- `large_batch_search.py`: 大batch size预训练搜索
- `zinc_hyperopt.py`: 通用ZINC数据集超参数搜索
- `analyze_optuna_results.py`: 结果分析工具
