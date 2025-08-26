# 超参数搜索目录

## 🎯 **大Batch Size专用搜索**（推荐）

针对高效的batch size [128, 256, 512]，内置已知最优结果作为起始点：

```bash
cd hyperopt/scripts

# 完整搜索（推荐）
python large_batch_search.py --bpe_mode all --pretrain_trials 20 --finetune_trials 15

# 只做预训练搜索  
python large_batch_search.py --stage pretrain --pretrain_trials 20

# 其他BPE模式
python large_batch_search.py --bpe_mode random --pretrain_trials 15
```

**特点**：
- ⏰ **高效**：专注于10-20分钟的快速batch size
- 🌱 **智能起始点**：内置9个最优参数组合，避免随机搜索
- 🎯 **实用**：平衡性能与训练速度

## 🔄 **完整搜索**

包含所有batch size范围 [16-1024]：

```bash  
python zinc_hyperopt.py --bpe_mode all --stage both
```

## 📊 **结果分析**

```bash
python analyze_optuna_results.py --bpe_mode all
python visualize_results.py
```

## 📁 **目录结构**

```
hyperopt/
├── scripts/
│   ├── large_batch_search.py    # 🎯 大batch专用（推荐）
│   ├── zinc_hyperopt.py         # 完整搜索  
│   ├── analyze_optuna_results.py # 结果分析
│   └── visualize_results.py     # 可视化
├── journal/                     # 数据库文件
└── results/                     # 分析结果
```

**推荐工作流**：使用 `large_batch_search.py` 进行高效搜索，获得实用的超参数配置。