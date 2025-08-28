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

## ✅ 当前发现与建议（基于本轮全组合测试）

- **序列级增强**
  - **swap**: 显著有效（预训练/微调均有益，微调阶段提升更稳定）。
  - **masking（仅微调阶段）**: 效果较为稳定且有益，可与swap叠加。
  - **deletion / truncation**: 平均来看对性能有害，不建议开启。
- **训练级增强**
  - **gaussian_noise**: 普遍略微有效，额外开销很小，推荐作为默认开关。
  - **consistency_regularization（R-Drop）**: 普遍有效，但需要双倍前向/反向，训练耗时约×2，调试阶段不建议开启，最终复现实验可考虑开启。
  - **feature_mixup**: 未观察到显著收益，不建议在当前设置下使用。

推荐配置（快速稳健）：微调阶段开启 `swap + masking + gaussian_noise`；如追求极致，可在最终跑加入 `consistency_regularization`（成本×2）。

## 📁 结果文件
- `results/pretrain_results.csv` - 预训练完整结果
- `results/pretrain_top_results.csv` - 预训练Top结果
- `results/finetune_results.csv` - 微调完整结果  
- `results/augmentation_effects.csv` - 增强方法效果分析

## ⚙️ 实验配置
- **数据集**: zinc（回归任务）
- **序列化**: feuler  
- **预训练**: 100 epochs（见生成脚本）
- **微调**: 30 epochs（见生成脚本）
- **目标属性**: homo

## 🔁 复现指南（我们测了什么、如何重现）

本目录覆盖了：
- 预训练阶段：序列级（deletion/swap/truncation）× 训练级（consistency）共16种组合；
- 微调阶段：序列级（deletion/swap/truncation/masking）× 训练级（consistency/noise/mixup）共128种组合；
- 结果收集与按增强项的效果统计分析。

### 1) 预训练（16种）
1. 生成命令（将写入 `pretrain_commands.sh`）
   ```bash
   python generate_pretrain_experiments.py
   ```
2. 执行全部预训练实验
   ```bash
   bash pretrain_commands.sh
   ```
3. 收集与分析（默认日志根目录为 `log/aug_pretrain/*`）
   ```bash
   python collect_pretrain_results.py --experiment_group aug_pretrain
   ```

说明：预训练增强通过 `config_json.bert.pretraining.mlm_augmentation_methods`（支持 `random_deletion/random_swap/random_truncation`）与 `augmentation_config.use_consistency_regularization` 控制。

### 2) 微调（128种）
1. 生成命令（将写入 `finetune_commands.sh`，并生成 `finetune_experiment_summary.txt`）
   ```bash
   python generate_finetune_experiments.py
   ```
2. 执行全部微调实验
   ```bash
   bash finetune_commands.sh
   ```
3. 收集与分析（将输出完整榜单与增强项效果统计）
   ```bash
   python collect_finetune_results.py --experiment_group aug_pretrain
   ```

说明：微调增强通过 `config_json.bert.finetuning.regression_augmentation_methods`（支持 `random_deletion/random_swap/random_truncation/sequence_masking`）与 `augmentation_config` 下的三个布尔开关控制：
- `use_consistency_regularization`（R-Drop）
- `use_gaussian_noise`
- `use_feature_mixup`

### 3) 快速复现实用示例
- 仅跑推荐组合（示例：基于 `P0100` 预训练，在微调阶段启用 `swap+masking+gaussian_noise`）：
  ```bash
  python run_finetune.py \
    --dataset zinc --method feuler \
    --experiment_group aug_pretrain --experiment_name F0101010 \
    --pretrain_exp_name P0100 --device auto --bpe_encode_rank_mode all \
    --epochs 30 --batch_size 512 --learning_rate 1e-4 \
    --config_json '"bert":{"finetuning":{"regression_augmentation_methods":["random_swap","sequence_masking"],"augmentation_config":{"use_gaussian_noise":true}}}}' \
    --plain_logs
  ```

## ⏱ 计算成本与实践建议
- 开启 R-Drop 会将训练时长约提升至 ×2，建议仅在最终复现实验开启；日常调试关闭。
- `gaussian_noise` 性价比高，建议默认开启；`mixup` 在当前设置下无明显收益。
- 若资源有限，优先级建议：`swap` > `masking（微调）` > `gaussian_noise` > 其他。
