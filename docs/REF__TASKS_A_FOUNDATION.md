# 基础治理任务（A类）

本文件聚焦于基础治理（A类）任务：配置统一化、实验命名与分组、日志与可视化。内容基于项目现状（`bert_classification.py`/`bert_regression.py`）与规范文档（`CONFIG_GUIDE.md`）。

## A1. 配置使用统一化

### 规范
- 项目统一从默认 `Config()` 实例创建；命令行仅最小覆盖：`--bs --lr --epochs --seed --name --group`
- 仅使用两大输出根目录：`logs/` 与 `model/`
  - 目录层级统一为：`<root>/<group>/<exp_name>/<dataset>/<method>/`
  - group：实验分组（支持多级，以路径层级表示）
  - exp_name：`{user_name}-{suffix}`，其中：
    - `user_name` 为输入指定的实验名称（可为空）
    - `suffix` 仅包含随机种子与简单时间戳（格式：`seed{n}-{mmdd_HHMM}`），不包含 `dataset/method` 或训练超参；是否启用 BPE 由末层 `method` 目录体现（`{serialization}-{BPE|RAW}`）
- 配置快照存放在 `logs/<group>/<exp_name>/<dataset>/<method>/`，命名：`<exp_id>_config.json`；其中 `<exp_id>` 作为文件名时会将 `/` 替换为 `_`（例如：`ablation/2025/try1-seed99-0808_1329` → `ablation_2025_try1-seed99-0808_1329_config.json`）
- 运行入口统一调用：
  - 若未指定 `--name`，在运行前生成默认实验名（见 A2）并赋值到 `config.experiment.experiment_name`
  - 调用 `config.experiment.setup_experiment(training=config.training)` 以生成 `experiment_id`
  - 构建目录：
    - `logs_dir = logs/<group>/<exp_name>/<dataset>/<method>`
    - `model_dir = model/<group>/<exp_name>/<dataset>/<method>`

### 项目现状与对齐
- 现状（参考 fine-tune 脚本）
  - 某些脚本存在 `output_dir/` 概念，内部包含 `logs/` 与 `results/`
  - 配置快照当前保存在 `output_dir/finetuning_config.json`
  - 模型路径通过 `config.get_bert_model_path()` 独立管理（常在 `model/` 下）
- 对齐计划（改造步骤）
  1. 在训练/评估脚本中，移除 `output_dir`，统一为：`logs_dir = logs/<group>/<exp_name>/<dataset>/<method>`，`model_dir = model/<group>/<exp_name>/<dataset>/<method>`
  2. 配置快照保存到：`logs_dir/<exp_id>_config.json`；不再写入 `output_dir/finetuning_config.json`
  3. 将原 `results/` 迁移到 `logs_dir/`（或子目录 `logs_dir/results/`）
  4. 封装/复用路径构建函数，避免分散写路径

### 验收
- 任一端到端脚本运行后，在 `logs/<group>/<exp_name>/<dataset>/<method>/` 可见 `<exp_id>_config.json` 与 TensorBoard/文本日志；在 `model/<group>/<exp_name>/<dataset>/<method>/` 可见检查点/权重

---

## A2. 实验命名与分组

### 规范
- `experiment_id = {group}/{exp_name}`
  - `exp_name = {user_name}-{suffix}`；当 `user_name` 为空时，`exp_name = {suffix}`
  - `suffix = seed{n}-{mmdd_HHMM}`（时间戳仅在 `exp_name` 中出现一次；BPE/RAW 通过 `method` 目录区分）
  - 指定 `--name`：保留 `user_name`，并在其后附加 `suffix` 作为后缀

### 项目现状与对齐
- 现状：
  - fine-tune 使用 `config.get_experiment_name(pipeline=...)` 获取实验名，并构造 `output_dir`
- 对齐计划：
  - `get_experiment_name` 内部兼容以上规则：
    - 无 name 时自动使用默认名
    - 有 name 时拼接默认片段以保证唯一性与可检索性

### 验收
- 不同方法/数据集/seed 并行实验，目录名可唯一区分且含关键信息（不依赖 `bs/lr`）；日志/快照/目录名一致

---

## A3. 日志与可视化

### 规范
- 统一使用 TensorBoard；默认开启，日志写入 `<output_dir>/logs/`
- 同步保存本地指标文件（JSON/CSV）到 `<output_dir>/results/`
- 保留 WandB 使用开关，但默认关闭

### 项目现状与对齐
- 现状：
  - 已通过 `SummaryWriter(log_dir=str(self.logs_dir))` 将 TensorBoard 日志写入 `logs/`
  - 文本日志写入 `logs/finetuning.log`
  - 结果保存于 `results/` 下（如 `test_results.json`、`finetuning_stats.json`）
- 对齐计划：
  - 统一在所有训练/评估脚本中复用相同的日志初始化与指标落盘工具

### 验收
- 任一训练脚本默认可写出 TensorBoard 日志与本地指标文件；输出位置与命名一致

---

## A5. 实施状态与用法示例（config重构完成）

### 状态（已完成，当前绿：22/22）
- 基于 `ProjectConfig` 的路径构建 API 已实现并落地到训练/微调脚本：`get_logs_dir` / `get_model_dir` / `ensure_experiment_dirs` / `get_config_snapshot_path`。
- 严格受控：未设置 `experiment_group` 抛错；不做静默回退。
- 旧路径接口已清理（统一迁移到新接口）：
  - 移除：`get_bert_log_dir`、`get_pretrained_model_path`、`get_bpe_model_path`、`get_gnn_model_path`、`get_gnn_log_dir`。
  - 调用方（训练/并行脚本等）已改为使用新接口。
- 新增/调整测试（均通过）：
  - `tests/test_config_foundation_a.py`、`tests/test_config_paths.py`
  - `tests/test_config_validation.py`（配置校验：heads 整除、device=auto 等）
  - `tests/test_paths_and_names.py`（exp_name、method_dir、快照命名）
  - `tests/test_wandb_metadata.py`（WandB 元数据字段）
  - `tests/test_effective_max_length.py`（序列有效长度上限/位置嵌入约束）
  - `tests/test_e2e_pretrain_finetune_qm9test.py`（真实 `qm9test` 预训练→微调端到端，日志/模型/结果产出齐全）

### 用法示例
```python
from config import ProjectConfig

cfg = ProjectConfig()
cfg.experiment_group = "ablation/2025"
cfg.experiment_name = "try1"  # 可为空
cfg.dataset.name = "qm9"
cfg.serialization.method = "graph_seq"
cfg.serialization.bpe.enabled = True
cfg.system.seed = 99

logs_dir = cfg.get_logs_dir()   # log/ablation/2025/try1-seed99-0808_1329/qm9/graph_seq-BPE
model_dir = cfg.get_model_dir() # model/ablation/2025/try1-seed99-0808_1329/qm9/graph_seq-BPE

# 配置快照建议写入：logs_dir / f"{exp_id}_config.json"
```

---

## A4. 可视化分组（TensorBoard / WandB）

### 规范
- TensorBoard：
  - 通过 `SummaryWriter(log_dir=logs/<group>/<exp_name>/<dataset>/<method>)` 使用分层目录，实现按 group/exp_name/dataset/method 的层次化浏览
  - 在 scalars/tag 中可补充 `{bpeFlag}`、`seed` 等标签，便于筛选
  - WandB（可选）：
    - `project` 固定
    - `group = "{group}/{exp_name}/{dataset}/{method}"`
    - `name = "{group}/{exp_name}__{dataset}/{method}"`
    - `tags = [dataset, method, ("BPE"|"RAW"), f"seed{seed}"]`

### 验收
- 打开 TensorBoard 时，能按目录层级自然分组到 dataset/method 下
- 启用 WandB 时，runs 在 UI 中按 group=`dataset/method` 分组，name 为 `exp_id`

---

## 附录：建议的工具函数/接口

- `default_experiment_name_from_training(training)`：可以用于生成可读名称片段（不强制）
- `format_exp_suffix(config)`：返回 `{dataset}_{method}-{bpeFlag}_seed{seed}` 片段
- `get_experiment_name(pipeline, name=None)`：实现 A2 规范；统一在训练/评估入口使用

---

*本文件不含时间指标；实施过程中所有规范变更应更新到 `CONFIG_GUIDE.md` 与相关脚本。*
