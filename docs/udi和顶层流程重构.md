## 数据-训练分层重构设计（提案）

### 为什么要改
- 重复构建与隐式耦合
  - 微调阶段通过 UDI 的 in-memory 路径再次触发序列化/BPE，与预训练重复，导致时间浪费与不确定性。
  - 训练层同时“读/构建数据”和“训练模型”，职责混杂，难以重用、难以测试。
- 配置与回退不透明
  - 训练时自动构建序列/压缩造成“隐式回退”，与科研代码“无 fallback”的规范相悖，且调试困难。
- 可复现性与确定性不足
  - 数据准备过程分散在多处（预训练、微调、UDI），难以确认哪一次构建产生的工件被使用。
- 可扩展性受限
  - 动态 BPE、并行多数据源、跨任务公用词表等需求，在当前耦合下很难实现。

### 我们的思想（分层、契约、无隐式回退）
- 单一职责分层
  - 数据准备层（构建层）：只负责“生成工件”（序列化、BPE、词表、分割、属性），可选择缓存；一次构建，多处复用。
  - 数据访问层（UDI）：只负责“统一访问工件”，提供 get/注册接口；不做任何构建/预处理/压缩。
  - 训练层（预训练/微调 Pipeline）：只消费“已准备好的工件”，执行 Dataset/DataLoader/Normalizer、模型与优化器构建、训练循环与评估；缺失工件直接报错。
- 契约优先
  - 明确定义训练层输入输出的工件契约（见下），保证不同脚本/阶段/方法间的互操作性。
- 无隐式回退
  - 训练层与 UDI 不做“若无则自动构建”。需要构建时，必须由外部准备器显式创建并注册。
- 以确定性驱动复现
  - 最顶层脚本“准备一次→在两阶段共享”；日志与工件明确可追溯（版本号/方法/bpe 状态/时间戳）。

### 目标
- 只序列化/BPE一次；端到端两阶段共享同一份结果
- 训练/微调 Pipeline 职责纯粹、易测、易复用
- UDI 成为严格的数据访问与元信息入口，不再承载构建
- 自动推断任务/标签（QM9 默认 homo；其他数据集使用 `default_target_property`），但仅在顶层准备器中执行
- 兼容现有脚本，逐步迁移，保留必要包装

---

## 新架构与接口

### 工件清单（Artifacts Contract）
- 数据：
  - train/val/test 序列（原始或 BPE 压缩）
  - split_indices（train/val/test 索引）
  - properties（与序列对齐的一致属性/标签来源）
  - vocab_manager（词表）
  - bpe_model（仅当使用 BPE）
- 元信息：
  - dataset_task_type（regression|classification）
  - default_target_property（回归默认属性键）
  - num_classes（分类类别数）

### UDI（数据访问层）
- 默认行为（关键）：
  - UDI 在创建时（未显式要求强制刷新）应尝试“加载当前可用的序列化与 BPE 工件”（若缓存存在则就绪）。
  - 若显式指定“强制刷新”或对应 processed 目录确无工件，则保持相应位置为空；一旦上层尝试读取，将直接抛错（不做隐式构建）。
- 能力边界：
  - UDI 只做“访问与状态报告”，不做任何构建（序列化/BPE/词表）。
  - UDI 应提供“状态查询/报告”接口，明确告知：哪些工件已就绪、哪些缺失（例如：has_serialized, has_bpe, has_vocab 等）。
- 接口分组：
  - 读取（保留）：get_sequences_cached(method)->get_sequences(method)、get_compressed_sequences_cached(method)->get_compressed_sequences(method)、get_split_indices()、get_downstream_metadata()、get_vocab()
  - 注册（新增）：register_serialized_sequences(method, sequences, properties, split_indices, version?)；register_bpe_compressed(method, compressed_sequences, bpe_model, properties, split_indices, version?)；register_vocab(vocab_manager)
  - 移除/弃用：get_sequences_in_memory()/get_compressed_sequences_in_memory()/get_training_data_in_memory() 以及任何“缺失则构建”的路径（已按本改动移除）

> 约定：训练层若通过 UDI 读取缺失工件，应立即报错；是否补全由上层脚本决定。

### 预训练训练层（Pretrain Pipeline）
- 函数（建议）：
  - pretrain(config, token_sequences, vocab_manager=None) -> {mlm_model, vocab_manager, stats}
  - build_vocab_from_tokens(token_sequences, config) -> vocab_manager
- 要点：
  - 必须由调用方传入 token_sequences（train/val/test 或统一合并后分配）；若 vocab_manager 为空，训练层仅“基于传入序列”构建词表，不再走任何数据加载/构建路径。

### 微调训练层（Finetune Pipeline）
- 函数（建议）：
  - finetune(task, udi, config) -> {model, metrics}
- 要点：
  - 训练层从 udi 读取 train/val/test 序列与标签（不构建）；若缺失则报错提示“请先使用准备器构建/注册工件”。

### 脚本形态与期望
- 预训练脚本（单阶段）
  - 期望：数据已准备好（UDI 创建后可直接读）。
  - 若 UDI 报缺失：直接报错并退出（不做补全），或由外部准备器先构建并 register_* 后再运行。
  - 标准流程：加载/验证 UDI 状态 → 调用 pretrain(...)（仅训练）。
- 微调脚本（单阶段）
  - 期望：数据已准备好（UDI 创建后可直接读）。
  - 若 UDI 报缺失：直接报错并退出（不做补全），或由外部准备器先构建并 register_* 后再运行。
  - 标准流程：加载/验证 UDI 状态 → 调用 finetune(...)（仅训练/评估）。
- 端到端脚本（两阶段）
  - 期望：强制刷新并在顶层“一次性准备”所有工件（序列化、BPE、划分、词表、属性等），完成后通过 register_* 写入 UDI；随后两阶段共享。
  - 标准流程：强制刷新 UDI → 顶层准备/注册 → pretrain(...) → finetune(...)

---

## 需要进行的改动（实施清单，更新）

- 数据访问层（UDI）
  - 删除/弃用：in-memory 构建方法与“缺失则构建”的任何路径
  - 新增：register_*（注册工件）接口
  - 保持：cached 读取与元信息读取
- 预训练训练层
  - 新增 API：pretrain(config, token_sequences, vocab_manager啊？)
  - 词表构建逻辑从内部提取为“基于传入 tokens 的独立函数”，训练层不得读取/构建数据。
- 微调训练层
  - 修改 API：finetune(task, udi, config)，只从 UDI 读取数据，缺失则报错（不构建）。
  - 任务/标签自动推断只在顶层准备器执行（QM9 默认 homo，其他数据集使用 default_target_property）。
- 顶层脚本
  - 预训练/微调（单阶段）：期望数据已就绪；如缺失，直接失败或先行准备再运行。
  - 端到端：强制刷新→一次准备→注册→两阶段共享，严禁二次构建。
- 配置
  - 默认 `task.target_property: null`
  - 顶层准备器按数据集元信息自动推断（QM9→homo；其他→default_target_property）

---

## 兼容策略
- 顶层脚本已接近最终形态，不再提供“旧式兼容路径”。若缺失工件，直接调整脚本或先行准备。

---

## 风险与应对
- 内存占用：in-memory 端到端注意数据规模（必要时切换为缓存）
- 迁移成本：API 变更需要改动调用处；通过包装与清晰错误信息降低成本
- 工件一致性：准备器需写入版本/方法/bpe_flag 标记，训练层日志中回显

---

## 验收标准
- 端到端（ZINC+feuler_bpe）仅发生一次序列化/BPE（日志中不再出现微调阶段的二次构建）
- ZINC 微调标签自动推断，无 KeyError
- 预训练/微调单阶段脚本均可在“只读已准备工件”模式下运行
- 端到端总时长缩短（去除重复构建）
- 日志中工件来源明确（dataset/method/BPE/version）

---

## 词表（VocabManager）设计更新
- 词表与“数据集”绑定，而非与“预训练模型”绑定：
  - 预训练模型训练时应加载该数据集对应词表；
  - 即使跳过预训练，微调也应能加载数据集对应词表并顺利跑通（性能另行评估）。
- 顶层准备器负责：从训练 tokens 统一构建或读取词表，并通过 UDI 注册；训练层仅消费。

---

## 后续里程碑（建议）
- M1：实现 register_* 接口；移除 UDI in-memory 构建；在端到端脚本打通“一次准备→两阶段共享”
- M2：重构预训练/微调 API；更新单阶段脚本
- M3：统一词表构建策略与落盘；完善元信息推断与验证
- M4：增加集成测试（ZINC/QM9/MNIST）

- 总结
  - 该设计将“构建数据”与“训练模型”彻底解耦，遵循无隐式回退、一次构建多次复用、可复现性的科研规范；解决了当前 in-memory 场景的重复序列化问题，并为动态 BPE、并发准备与跨任务共享打下基础。