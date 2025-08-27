# 超参搜索与微调流水线（权威说明）

本文件描述项目内从预训练、结果分析、候选参数生成到微调超参搜索的完整、可复现流程；所有命令均以仓库根目录为工作目录执行。

## 目录规范与命名约定

- 预训练与微调模型目录：`model/<experiment_group>/<experiment_name>/`，最佳权重位于 `best/`
- 搜索数据库（Optuna Journal）：`hyperopt/journal/*.db`
- 分析/候选输出：`hyperopt/results/*.json|*.csv`
- 统一组名与实验名是后续流程能否正确找到模型的关键（必须显式指定）。

命名约定（示例）：
- 组名（group）：`large_bs_hyperopt_all`
- 预训练默认实验名（default_config）：`large_bs_all_pt_default`
- 大批量超参搜索实验名（trial 编号式）：`large_bs_all_pt_000`、`..._001`、`..._002`

---

## 阶段A：单模型预训练（权威入口）

脚本：`run_pretrain.py`（必须参数：`--dataset --method --experiment_group --experiment_name`）

示例1：训练“默认参数预训练模型”（供微调候选中的 default_config 使用）
```bash
python run_pretrain.py \
  --dataset qm9 \
  --method feuler \
  --experiment_group large_bs_hyperopt_all \
  --experiment_name large_bs_all_pt_default \
  --bpe_encode_rank_mode all
```

示例2：显式指定基础超参（其余用默认）
```bash
python run_pretrain.py \
  --dataset qm9 --method feuler \
  --experiment_group large_bs_hyperopt_all \
  --experiment_name large_bs_all_pt_default \
  --bpe_encode_rank_mode all \
  --epochs 200 --batch_size 256 --learning_rate 4e-4
```

可选：复杂嵌套配置用 `--config_json`（JSON 字符串或文件路径），不要使用其他“覆盖器”。

---

## 阶段B：大批量预训练搜索（可选）

脚本（示例）：`hyperopt/scripts/large_batch_search.py` 或 `hyperopt/scripts/zinc_hyperopt.py`

- 这类脚本会创建 Optuna Study 并批量运行预训练试验，训练结束会将 trial→模型路径信息保存到各自的模型目录和 Journal 中。
- 生成的 Journal 示例：`hyperopt/journal/large_batch.db`
- 典型预训练实验名格式：`large_bs_all_pt_{trial:03d}`（组名：`large_bs_hyperopt_all`）

---

## 阶段C：分析与候选参数提取

1) 快速浏览与统计
```bash
python hyperopt/scripts/analyze_optuna_results.py \
  --journal hyperopt/journal/large_batch.db \
  --bpe_mode all
```

2) 生成微调阶段的“预训练选项”集合（Top3 overall + 各方法Top1 + default_config）
```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/large_batch.db \
  --target_study methods_large_batch_pretrain_all \
  --output_dir hyperopt/results
```

输出：
- `hyperopt/results/best_pretrain_params_for_finetuning.json`
  - 包含每个候选的 `method`、`bs/lr/wd/...` 以及定位 trial 的关键信息（如 loss）
  - `default_config` 项仅作为“预训练实验名”约定（需要先完成阶段A的默认模型训练）

---

## 阶段D：基于候选预训练模型的微调超参搜索（权威流程）

脚本：`hyperopt/scripts/finetune_with_pretrain_options.py`

核心原则：
- “预训练选项”只用于决定“加载哪个预训练模型”以及采用哪个 `method`
- 微调阶段的超参数（`lr/bs/wd/grad_norm/warmup_ratio/epochs`）全部由 Optuna 重新搜索
- 评估指标：严格使用测试集 MAE 作为唯一优化目标

运行（使用脚本内默认参数）：
```bash
python hyperopt/scripts/finetune_with_pretrain_options.py
```

注意：
- 若选中 `default_config`，脚本将按约定路径查找：
  - 组：`large_bs_hyperopt_all`
  - 名：`large_bs_all_pt_default`
  - 路径：`model/large_bs_hyperopt_all/large_bs_all_pt_default/best/`
- 其他候选会通过 Journal（如 `hyperopt/journal/large_batch.db`）定位对应 trial，并构造标准实验名 `large_bs_all_pt_{trial:03d}` 与组 `large_bs_hyperopt_all` 进行加载。

---

## 阶段E：单次微调（对齐 commands.list 的用法）

脚本：`run_finetune.py`

按“组 + 预训练实验名”加载模型：
```bash
python run_finetune.py \
  --dataset qm9 \
  --method feuler \
  --experiment_group large_bs_hyperopt_all \
  --pretrain_exp_name large_bs_all_pt_default \
  --bpe_encode_rank_mode all
```

也可显式指定目录：
```bash
python run_finetune.py \
  --dataset qm9 \
  --method feuler \
  --pretrained_dir model/large_bs_hyperopt_all/large_bs_all_pt_default/best \
  --bpe_encode_rank_mode all
```

---

## 阶段F：批量触发微调搜索

脚本：`runs/search_ft.sh`

- 内部多次调用 `python hyperopt/scripts/finetune_with_pretrain_options.py`
- 固定参数已写为脚本默认，不从命令行传入，避免分散配置

---

## 关键约束与坑点

- 实验组织：务必统一 `experiment_group` 与 `experiment_name`；后续所有加载依赖这对键。
- Journal 一致性：用于从 trial→实验名→模型路径的映射，换库或换 study 名时需同步更新分析脚本的 `--target_study`。
- default_config：只是一个“命名约定”，需要先训练一次默认预训练模型；否则微调脚本会跳过或报错。
- 评估指标：微调搜索严格用测试集 MAE；无回退分支（缺失则报错）。

---

## 参考脚本清单

- 训练/微调入口
  - `run_pretrain.py`（单次预训练权威入口）
  - `run_finetune.py`（单次微调权威入口）

- 批量与搜索
  - `batch_pretrain_simple.py`（并行批量调用 run_pretrain.py，不改变参数语义）
  - `hyperopt/scripts/large_batch_search.py`、`hyperopt/scripts/zinc_hyperopt.py`（Optuna 搜索）

- 结果分析与候选生成
  - `hyperopt/scripts/analyze_optuna_results.py`
  - `hyperopt/scripts/extract_best_params_for_finetuning.py`

---

## 最小端到端流程（可直接拷贝）

1) 训练默认预训练模型（一次性）
```bash
python run_pretrain.py \
  --dataset qm9 --method feuler \
  --experiment_group large_bs_hyperopt_all \
  --experiment_name large_bs_all_pt_default \
  --bpe_encode_rank_mode all
```

2) 从大批量搜索产物中提取候选
```bash
python hyperopt/scripts/extract_best_params_for_finetuning.py \
  --journal hyperopt/journal/large_batch.db \
  --target_study methods_large_batch_pretrain_all \
  --output_dir hyperopt/results
```

3) 运行微调超参搜索（MAE 目标）
```bash
python hyperopt/scripts/finetune_with_pretrain_options.py
```


