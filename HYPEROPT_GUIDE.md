# ZINC数据集超参数搜索使用指南

## 📋 概述

为ZINC数据集实现的两阶段超参数搜索系统：
1. **预训练阶段**：搜索预训练超参数，保存top-k个最优模型
2. **微调阶段**：基于最优预训练模型搜索微调超参数

支持多节点分布式搜索，使用Optuna + Journal存储。

## 🎯 搜索的参数

### 连续参数 (对数/线性分布)
- **learning_rate**: [1e-5, 1e-3] (对数分布)
- **weight_decay**: [0.0, 0.3] (线性分布) 
- **max_grad_norm**: [0.0, 3.0] (线性分布)
- **warmup_ratio**: [0.0, 0.3] (线性分布，学习率warmup比例)
- **mask_prob**: [0.10, 0.25] (线性分布，仅预训练)

### 离散参数
- **batch_size**: [16, 32, 64, 128, 256, 512, 1024] (二的幂次，可通过--max_batch_size限制)
- **bpe_encode_rank_mode**: ['none', 'all', 'random', 'gaussian'] (分别搜索)

### 固定参数
- **encoder_type**: 'gte' (性能更优)
- **serialization_method**: 'fcpp' (固定使用，性能较好的C++实现)
- **bpe_num_merges/min_frequency**: 数据预处理时固定

## 📚 参数说明

### Warmup Ratio
**作用**: 控制学习率线性预热的比例
- `warmup_ratio = 0.0`: 无warmup，直接使用目标学习率
- `warmup_ratio = 0.1`: 前10%的训练步数进行线性warmup
- `warmup_ratio = 0.3`: 前30%的训练步数进行线性warmup

**原理**: 训练开始时学习率从0线性增加到目标值，帮助模型稳定收敛，特别对大学习率有效。

**搜索范围理由**: [0.0, 0.3]覆盖了从无warmup到较长warmup的范围，超过30%通常没有必要。

## 🚀 使用方法

### 1. 单个BPE模式搜索

```bash
python zinc_hyperopt.py \
    --bpe_mode all \
    --journal_file "./journals/zinc_all.journal" \
    --stage both \
    --pretrain_trials 50 \
    --finetune_trials 100 \
    --top_k 5 \
    --max_batch_size 512  # 限制最大batch size避免显存溢出
```

### 2. 批量搜索所有BPE模式

```bash
chmod +x run_zinc_search.sh
./run_zinc_search.sh
```

### 3. 仅运行预训练阶段

```bash
python zinc_hyperopt.py \
    --bpe_mode all \
    --journal_file "./journals/zinc_all.journal" \
    --stage pretrain \
    --pretrain_trials 50
```

### 4. 仅运行微调阶段（需要先有预训练结果）

```bash
python zinc_hyperopt.py \
    --bpe_mode all \
    --journal_file "./journals/zinc_all.journal" \
    --stage finetune \
    --finetune_trials 100 \
    --top_k 5
```

## 📊 查看结果

### 查看所有模式的结果
```bash
python view_results.py --journal_dir ./journals
```

### 查看特定BPE模式结果
```bash
python view_results.py --journal_dir ./journals --bpe_mode all
```

## 🌐 多节点分布式搜索

### 方法1: 共享文件系统
在多个节点上运行相同的命令，指向同一个journal文件：

```bash
# 节点1
python zinc_hyperopt.py --bpe_mode all --journal_file "/shared/zinc_all.journal" --stage both

# 节点2 (同时运行)
python zinc_hyperopt.py --bpe_mode all --journal_file "/shared/zinc_all.journal" --stage both
```

### 方法2: 分模式并行
不同节点搜索不同的BPE模式：

```bash
# 节点1: none + all
python zinc_hyperopt.py --bpe_mode none --journal_file "./journals/zinc_none.journal" --stage both
python zinc_hyperopt.py --bpe_mode all --journal_file "./journals/zinc_all.journal" --stage both

# 节点2: random + gaussian  
python zinc_hyperopt.py --bpe_mode random --journal_file "./journals/zinc_random.journal" --stage both
python zinc_hyperopt.py --bpe_mode gaussian --journal_file "./journals/zinc_gaussian.journal" --stage both
```

## ⚙️ 核心特性

### 1. Optuna剪枝支持
- 预训练：基于验证损失剪枝低效试验
- 微调：基于主要指标(MAE/RMSE)剪枝

### 2. 实验名称管理
自动生成不重复的实验名称：
- 预训练：`zinc_all_pt_001_1201_1430`
- 微调：`zinc_all_ft_001_005_1201_1430`

### 3. 模型保存策略
- 预训练：保存top-k个最优模型
- 微调：基于最优预训练模型进行微调

### 4. 错误处理
- 训练失败自动触发`TrialPruned`
- 支持中断后恢复搜索

## 📁 输出结构

```
├── journals/
│   ├── zinc_none.journal       # None模式搜索记录
│   ├── zinc_all.journal        # All模式搜索记录  
│   ├── zinc_random.journal     # Random模式搜索记录
│   └── zinc_gaussian.journal   # Gaussian模式搜索记录
├── model/
│   ├── zinc_all_pt_001_*/      # 预训练模型
│   └── zinc_all_ft_001_005_*/  # 微调模型
└── logs/
    └── ...                     # 训练日志
```

## 🔧 参数调优建议

### 试验数量
- **预训练**: 50-100次 (搜索空间较大)
- **微调**: 100-200次 (基于top-k模型，搜索空间相对较小)

### Top-K选择
- **K=3**: 快速探索
- **K=5**: 平衡性能与计算成本  
- **K=10**: 充分利用预训练多样性

### Batch Size调优
- **16GB显存**: 推荐 `--max_batch_size 256`
- **24GB显存**: 推荐 `--max_batch_size 512`
- **32GB+显存**: 可使用默认 `--max_batch_size 1024`

**注意**: 大batch size通常需要更高学习率和更长warmup。

### 超时设置
```bash
# 设置24小时超时
python zinc_hyperopt.py --timeout 86400 ...
```

## 🔄 并发优化机制

### Constant Liar策略
为避免多进程生成重复超参数，系统采用Optuna的Constant Liar策略：

```python
# 并发场景示例：
进程A: 开始trial_001 {lr=1e-4, bs=32} -> "正在运行"
进程B: 读取历史 -> 看到trial_001正在运行 -> 假设其结果为"谎言值"
进程B: 基于假设结果采样 -> 选择不同参数 {lr=2e-4, bs=64}
进程A: 完成trial_001 -> 真实结果替换"谎言值"
```

### 关键配置
- **seed=None**: 不固定随机种子，避免相同随机序列
- **constant_liar=True**: 启用占位策略，减少参数重复
- **JournalStorage**: 多进程共享试验历史

## ⚠️ 注意事项

1. **磁盘空间**: 每个模型约几百MB，注意预留足够空间
2. **内存使用**: GTE模型需要较大显存，建议16GB+
3. **并发控制**: 建议每张GPU运行一个进程，避免显存竞争
4. **数据准备**: 确保ZINC数据集已正确预处理
5. **网络环境**: 首次运行会下载GTE预训练权重

## 🐛 故障排查

### 常见错误
1. **CUDA内存不足**: 减小batch_size搜索范围
2. **数据集未找到**: 检查ZINC数据集路径
3. **Journal文件损坏**: 删除后重新开始搜索

### 调试模式
```bash
# 减少试验数进行测试
python zinc_hyperopt.py --pretrain_trials 3 --finetune_trials 5 ...
```
