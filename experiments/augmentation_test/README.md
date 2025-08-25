# 数据增强实验框架

这个目录包含了完整的数据增强实验设计，分为两个阶段：

## 🔬 实验流程

### 阶段1：预训练筛选 (16种组合)
1. 生成预训练实验：
   ```bash
   python generate_pretrain_experiments.py
   ```

2. 执行预训练实验：
   ```bash
   bash pretrain_commands.sh
   ```

3. 收集和分析结果：
   ```bash
   python collect_pretrain_results.py --experiment_group aug_pretrain
   ```

### 阶段2：微调详测 (128种固定组合)
1. 生成全部128种微调实验：
   ```bash
   python generate_finetune_experiments.py
   ```

2. 执行微调实验：
   ```bash
   bash finetune_commands.sh
   ```

3. 收集和分析结果：
   ```bash
   python collect_finetune_results.py --experiment_group aug_pretrain
   ```

## 📊 实验设计

### 预训练增强 (16种组合)
- **序列级**: deletion, swap, truncation (3种)
- **训练级**: consistency_regularization (1种)  
- **编码**: P{seq3位}{train1位}，如P1011表示deletion+truncation+consistency

### 微调增强 (128种固定组合)
- **序列级**: deletion, swap, truncation, masking (4种)
- **训练级**: consistency_regularization, gaussian_noise, feature_mixup (3种)
- **编码**: F{seq4位}{train3位}，如F10110101
- **映射关系**: 每8个微调实验对应1个预训练实验（前4位的前3位+训练级第1位）

## 📁 结果文件
- `results/pretrain_results.csv` - 预训练完整结果
- `results/pretrain_top_results.csv` - 预训练Top结果
- `results/finetune_results.csv` - 微调完整结果  
- `results/augmentation_effects.csv` - 增强方法效果分析

## ⚙️ 实验配置
- **数据集**: zinc (回归任务)
- **序列化**: feuler  
- **预训练**: 20 epochs
- **微调**: 30 epochs
- **目标属性**: homo
