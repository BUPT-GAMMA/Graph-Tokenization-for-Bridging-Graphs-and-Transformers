# 任务计划（不含时间指标）

本文档描述本项目下一阶段需要完成的重构与实现任务，按模块组织，说明要做什么、产出物及验收标准。所有任务遵循项目规则：单一配置源、无隐式回退、使用真实数据、结果可复现、以测试与基准为先。

## 0. 目标与通用原则
- 统一配置与实验管理，减少分散参数与不确定性来源
- 数据处理与划分完全确定性，序列化/BPE 输出在等价实现间保持一致
- 优化性能前先建立基准与正确性对齐测试；并发需可验证地带来收益
- 文档与代码同步更新；任何核心接口变更应推动全局统一迁移

---

## A. 基础治理

### A1. 配置使用统一化（已完成，已通过测试验证，22/22）
- 任务
  - 项目统一从默认 `Config()` 实例创建；命令行仅最小覆盖 `--bs --lr --epochs --seed --name --group`
  - 每次实验保存“实际使用的配置快照”到标准位置：
    - 主存档：`<output_dir>/<exp_id>_config.json`
    - 便于排查：在 `<output_dir>/logs/` 下生成同名符号链接或副本（与日志同目录，便于关联查阅）
  - 对现状的总结（基于 `bert_classification.py` / `bert_regression.py`）：
    - 目录结构：`output_dir/` 下包含 `logs/`（文本日志+TensorBoard）与 `results/`（评估结果）
    - 配置快照：当前保存在 `output_dir/finetuning_config.json`
    - 模型：通过 `config.get_bert_model_path()` 生成的路径独立管理（通常在 `model/` 下）
  - 对齐计划：统一约定配置快照位于 `<output_dir>/<exp_id>_config.json`，并在 `logs/` 下提供符号链接（兼容现状，便于定位）
- 产出
  - 统一配置入口与覆盖逻辑落地，训练/微调脚本与并行脚本均已改造
- 验收（均已通过）
  - 目录层级：`log|model/<group>/<exp_name>/<dataset>/<method>`（`method = {serialization}-{BPE|RAW}`）
  - `exp_name`：`{user_name}-{seedN-mmdd_HHMM}`；未提供 `user_name` 时为 `seedN-mmdd_HHMM`
  - 配置快照：写入日志目录根层，文件名 `<exp_id>_config.json`（将 `/` 替换为 `_`）
  - WandB 元数据：`group="{group}/{exp_name}/{dataset}/{method}"`，`name="{group}/{exp_name}__{dataset}/{method}"`，`tags=[dataset, method, BPE|RAW, seed]`
  - 训练/微调：端到端用例（真实 `qm9test`）通过，产物结构符合规范
  - 详见 `docs/TASKS_A_FOUNDATION.md` 更新

### A2. 实验命名与分组（已完成，已通过测试验证）
- 任务
  - 无显式名称时，用训练配置生成默认名（`bs{batch}_lr{lr}`），ID=`{group}/{name}_{timestamp}`
    - 具体命名可参考 `CONFIG_GUIDE.md` 的 `default_experiment_name_from_training` 与现有 `config.get_experiment_name(...)` 用法
    - 为提高可检索性，建议默认名附加关键信息：`{dataset}_{serialization}-{bpeFlag}_seed{seed}`（示例：`bs32_lr1e-4__qm9_graphseq-BPE_seed42`）
- 产出
  - 规范化的实验目录/日志/指标命名
- 验收（均已通过）
  - 不指定 `--name` 时：自动生成可读实验名，并在日志/快照/目录保持一致
  - 指定 `--name` 时：保留用户自定义 name，并追加默认可读配置片段用于区分（如：`{name}__{dataset}_{method}-{bpeFlag}_seed{seed}`），确保并行不同方法/数据集的实验目录可唯一辨识
  - 测试覆盖：`tests/test_config_paths.py`、`tests/test_paths_and_names.py`

### A3. 日志与可视化（已完成，已通过测试验证）
- 任务
  - 统一使用 TensorBoard；保留本地 JSON/CSV 指标文件
  - 保持 WandB 的可选开关，但默认关闭
- 产出
  - 标准化的日志初始化与落盘工具
- 验收（均已通过）
  - 训练/微调脚本写出 TensorBoard 日志到标准 `logs_dir`
  - 本地结果文件写到 `logs_dir/results/`
  - WandB 元数据接口对齐规范（`tests/test_wandb_metadata.py`）
  - 端到端用例（`tests/test_e2e_pretrain_finetune_qm9test.py`）验证日志与产物结构







---

## B. 数据层（进度同步）

### B1. 固定划分与确定性（已完成，已通过测试验证）
- 任务
  - 保持现有的简单数据结构：单一 `data.pkl` + 三个索引文件 `train_index.json`、`val_index.json`、`test_index.json`
  - loader 仅按索引加载合并，不在内部随机划分；缺失文件直接报错，不做回退
- 产出
  - `data/<dataset>/data.pkl` 与三份索引 JSON 的加载逻辑
- 验收
  - 多次运行得到完全一致的数据划分；禁用内部随机切分；缺失文件时显式报错
  - 现状说明（代码已对齐）：
    - 统一索引读取：`src/data/base_loader.py:get_split_indices()`；异常即抛 `FileNotFoundError`
    - 各数据集 loader 按索引切分：`src/data/qm9_loader.py`、`src/data/qm9test_loader.py`、`src/data/zinc_loader.py`、`src/data/aqsol_loader.py`、`src/data/mnist_loader.py`
    - 无内部随机切分与隐式回退
  - 测试覆盖：`tests/8.9_refactor/taskB/test_unified_data_interface_errors.py`、端到端用例均以真实 `qm9test` 运行

### B2. 统一数据接口 UnifiedDataInterface（基本完成：核心接口已完成；版本发现/选择待实现）
- 任务
  - 设计并实现：
    - `get_graphs()`
    - `get_sequences(version: str = "latest")`
    - `get_compressed_sequences(bpe_model: str)`
  - 支持数据版本/缓存发现与选择；面向多数据集扩展
- 产出
  - `src/data/unified_data_interface.py`（或在现有结构内实现）
- 验收
  - 序列化/BPE/训练可通过同一接口取数；切换数据版本无需改业务代码
  - 现状说明：
  - 已实现统一入口：`get_graphs()`、`get_sequences(method, build_if_missing)`、`get_compressed_sequences(method, build_if_missing)`、`get_all_sequences(...)`、按 split 获取接口、`get_training_data(...)`
  - 路径规范：结果写入 `processed_data_dir/<dataset>/serialized_data/<method>/serialized_data.pickle` 与 `.../bpe_compressed/<method>/compressed_data.pickle`
  - 错误策略：默认缺失即抛错；仅在 `build_if_missing=True` 时进行确定性构建，无隐式回退
  - 待办：`data_version` 目前为占位参数，需扩展为版本化子目录或哈希命名并提供发现 API 与测试
  - 测试覆盖：`tests/8.9_refactor/taskB/test_unified_data_interface.py`、`test_udi_new_methods.py`、`test_unified_data_interface_errors.py`、`test_udi_get_training_data.py`

### B3. 工厂与职责边界（已完成）
- 任务
  - `DataFactory` 负责选择 loader；数据集特定 token 映射由 data 层暴露，serializer 仅读取
- 产出
  - 明确的工厂创建路径与示例
- 验收
  - 修改或新增数据集时，无需改 serializer 接口
  - 现状说明：
    - 工厂集中注册与创建：`src/data/unified_data_factory.py`
    - 数据层暴露 token 相关接口：各 `*_loader.py` 的 `get_node_token/get_edge_token/get_token_map/get_*_type`
    - 序列化器通过数据层取 token：`src/algorithms/serializer/base_serializer.py` 中 `get_node_token/get_edge_token`

---

## C. 序列化算法层（进度同步）

### C1. 接口与命名统一（基本完成）
- 任务
  - 全量实现/统一以下方法：
    - `serialize(graph)` / `batch_serialize(graphs)`
    - `multiple_serialize(graph)` / `batch_multiple_serialize(graphs)`
— 产出
  - 标准基类与具体实现一致的签名
— 验收
  - 任意序列化器均能以统一方式被上层调用
— 现状说明
  - `serialize/batch_serialize/multiple_serialize/batch_multiple_serialize` 已在主力方法对齐，可用于上层统一调用；后续新增方法按同一命名规范接入

### C2. 确定性与多数据集兼容（基本完成）
- 任务
  - 在多数据集场景下保证确定性输出；如需数据集特定信息，通过 data 层提供
- 产出
  - 文档化的依赖项与读取方式
— 验收
  - 同一图在同一配置下输出一致；跨数据集不会出现未定义分支
— 现状说明
  - 依赖数据层提供 token/类型信息；当前在多数据集路径上均走显式分支，无隐式回退

### C3. 批处理并行（暂缓：性能优化将在端到端跑通后推进）
- 任务
  - 为 `batch_serialize`/`batch_multiple_serialize` 提供可选并行实现
  - 附带性能自检：小样本串行/并行对比，无收益则回退串行
— 产出（暂缓）
  - 并行开关参数与性能检测逻辑
— 验收（暂缓）
  - 在目标规模数据上并行可验证带来加速，且输出与串行一致

---

## D. BPE 压缩（进度同步）

### D1. 核心实现优化（向量化/替代编码）
- 任务
  - 参考 minBPE，对 encode 引入替代实现；保持训练过程不变，保留等价行为
- 现状说明
  - 训练流程保持既有实现；在 `main_bpe.py` 中提供 minBPE 风格的 `encode` 备选路径（小序列优先），短期未见显著提速，作为 alternative 选项保留
- 待办
  - 将 encode 策略作为显式配置开关；补充 encode/decode 等价性测试与可逆性检查；评估 `minbpe_ids.py` 路径的一致性
- 验收（暂定）
  - 等价性测试稳定通过；在真实序列上可重复的速度收益（若收益不足，记录基线并维持现状）

### D2. 并行与缓存（暂缓）
- 任务（暂缓）
  - 支持 encode/decode 与 train 的并行；制定缓存规范
- 产出（暂缓）
  - 并行开关、缓存规范、持久化路径规则
- 验收（暂缓）
  - 多进程/多线程下结果一致；缓存不串扰、可清理

### D3. 基准与对齐（需完善）
- 任务
  - 完善端到端基准与等价性测试
  - 审视并改造项目根目录的 `run_serialization_bpe_comparison_simple.py`：
    - 统一仅输出结构化数据（JSON/CSV），图表与 Markdown 报告改为可选
    - 校验方法列表与现实现状一致；提供明确的失败原因与退出码
    - 支持通过统一配置读取参数（避免硬编码）
- 产出
  - `benchmarks/bpe_benchmark.py`、`tests/test_bpe_equivalence.py`、经简化的比较脚本
- 验收
  - 基准可重复；等价性测试稳定通过；比较脚本在 `qm9test` 上默认可运行并产出结构化指标

---

## E. 训练与评测（下一阶段重点）

### E1. 训练流水线标准化（未完成 → 下一步推进）
- 近期目标
  - 打通最小可用端到端：图数据序列化 →（可选）BPE → BERT 预训练 → 微调
  - 全面使用 UDI 取数；统一配置、命名与日志落盘；默认关闭 WandB
- 产出
  - 标准训练入口与公共工具
- 验收
  - `qm9test` 上端到端用例稳定跑通；产物结构与配置快照符合规范

### E2. 任务头模块化（未完成）
- 任务
  - 标准化 MLP 头/Pooling 策略定义与注册
- 产出
  - 任务头模块与接口说明
- 验收
  - 更换任务头无需改训练主干

### E3. 指标模块完善（未完成）
- 任务
  - 回归：MAE/MSE/RMSE/R²/Pearson/Spearman
  - 分类：Accuracy/Precision/Recall/F1（必要时 AUC/AP）
  - 统一导出 CSV/TSV 表格
- 产出
  - `src/utils/metrics.py`（或等价路径）
- 验收
  - 任一任务的评估输出可直接用于对比汇总

---

## F. 缓存与并发策略（侧重架构说明）

### F1. 缓存
- 任务
  - 优先文件系统缓存；仅对明确瓶颈函数使用 LRU，并提供清理工具
- 产出
  - 缓存管理工具与命名规范
- 验收
  - 缓存命中可观、命名可追踪、可一键清理

### F2. 并发
- 任务
  - 并发前先建立性能基准；提供死锁/串行化/收益不足检测
- 产出
  - 并发护栏与回退策略（非功能回退，而是性能决策）
- 验收
  - 并发开启后性能与资源使用符合预期；关闭后功能完全一致

### F3. 数据流/架构文档（新增）
- 任务
  - 产出一份简洁、准确、易懂的端到端数据流与数据结构说明（从原始图→序列化→BPE→训练/评测）
- 产出
  - `docs/DATA_FLOW.md`（或等价文件），覆盖路径规范、缓存位置、关键接口
- 验收
  - 新人可据此快速理解并复现实验；与现有实现一致、无歧义

---

## G. 测试体系（随重构同步推进）

### G1. 单元测试（底层优先）
- data：加载/格式/范围检查（图/标签）、固定划分一致性
- serializer：重构图一致性、点边数验证、无非法边/路径
- bpe：可逆性测试、与旧版等价性测试

### G2. 集成测试
- 数据→序列化→BPE→模型输入的串联正确性

### G3. 端到端测试
- 以真实数据（如 qm9test）跑通训练与评估；固定种子确保确定性
- 标记为 `e2e`，默认不运行，需显式指定（保持测试时间可控）

### G4. 性能测试
- BPE/序列化模块级基准与端到端基准；基准脚本固定输入与打印关键信息

---

## H. 文档与清理（随重构同步推进）

### H1. 文档
- 完成并维护：`DATA_GUIDE.md`、`MODEL_GUIDE.md`、`SETUP.md`
- 修改即更新文档；核心接口变更需写清迁移方式

### H2. 旧实现清理与备份
- BPE 历史版本：打包备份到 `backup/`，保留说明与迁移指南
- 删除过度工程化模块（插件/部署/安全等），以科研代码简洁为先

---

## 非目标（Non-Goals）
- 不在此项目内提供服务化部署/REST API
- 不在此项目内实现 GNN 主体（可在其他仓库进行）
- 不产出“结论性报告”，仅产出可复用数据/日志/指标（后续分析使用）

## 依赖与前置条件
- 真实数据可用且版本明确；数据预处理脚本可生成稳定 splits
- 配置文件与路径软链接方案在当前环境下有效

## 风险与对策（简述）
- 并发导致卡死或无收益：基准先行；自动检测收益不足→串行
- 向量化带来行为差异：先做等价性测试；必要时分阶段替换
- 配置散落：统一入口与覆盖，禁止在子模块动态改配置

---

*本文档不包含时间指标；任务将按优先级推进，并在实现过程中保持文档与代码的持续同步更新。*
