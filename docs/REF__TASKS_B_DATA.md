## 数据层任务（B类）

本文档在项目既有规范基础上，结合当前代码实现，给出“数据层（B类）”的目标、现状评估、改造方案与验收标准。目标与风格对齐 `docs/TASK_PLAN.md` 与 `docs/TASKS_A_FOUNDATION.md`，遵循项目通用原则：单一配置源、无隐式回退、不使用假数据、结果可复现、接口统一、职责边界清晰。

---

### B0. 范围与目标（与 TASK_PLAN 对齐）

- 固定数据划分与确定性：保持 data/<dataset>/ 下的三份索引文件；加载端仅按索引取数，不做内部随机切分。
- 统一数据接口 UnifiedDataInterface：以统一入口提供 graphs / sequences / compressed sequences 的读取与版本管理，屏蔽底层路径细节。
- 工厂与职责边界：由 `DataFactory`（统一工厂）选择具体 loader；“数据集特定 token 映射/类型信息”由 data 层提供，serializer 仅读取，不自行猜测。

---

### B1. 现状评估（已按回退决策调整）

基于代码快速扫描（关键文件位于 `src/data/`、`data_prepare.py`、`config.py`）：

- 固定划分与确定性（splits）
  - 现状：
    - `BaseDataLoader.get_split_indices()` 与 `UnifiedDataInterface._load_split_indices()` 从 `data/<dataset>/train_index.json|val_index.json|test_index.json` 读取索引；若缺失则抛错，无随机切分。
    - 各具体 loader 在 `_load_processed_data()` 中按上述索引进行切分，读取 `data.pkl`。示例见 `src/data/qm9_loader.py`。
  - 已修正：
    - `BaseDataLoader.get_split_indices()` 的异常信息统一为通用表述。

- 统一数据接口（UnifiedDataInterface）
  - 现状：
    - 存在 `src/data/unified_data_factory.py`（统一工厂，注册多数据集 loader）与若干“便捷接口”函数 `src/data/quick_interface.py`（如 `get_bert_training_data`）。
    - 尚未有显式的 `UnifiedDataInterface` 类，未提供统一的 graphs/sequences 读取接口与数据版本选择能力。
  - 积极点：
    - `quick_interface.get_bert_training_data(...)` 已体现出用“loader 的 split 索引 + 序列化/BPE 缓存”构造训练数据的思路，具备可迁移到统一接口的基础。

- 工厂与职责边界
  - 现状：
    - 统一工厂 `UnifiedDataFactory` 已实现与注册（`qm9`、`qm9test`、`zinc`、`aqsol`、`mnist`）。
    - 各 loader 提供“数据集特定 token 映射/类型信息”，例如 `QM9Loader.get_node_token/get_edge_token/get_token_map`，serializer 在流水线中通过 `initialize_with_dataset(dataset_loader, graphs)` 使用，职责划分基本符合规划。
  - 问题：
    - `BaseDataLoader` 中列出的固定比例常量仅用于元数据标注，应确保不会被误用为运行时切分来源（当前实现已避免，但需在文档与测试中明确）。

---

### B2. 设计与实施方案（保持原有存储结构，不新增 splits 层级）

遵循：
- 不在 loader 内部做随机切分；splits 必须由独立步骤离线生成，并以文件固化。
- 单一配置源：路径/版本/seed 等从 `ProjectConfig` 读取；禁用隐式回退。
- 对外提供统一接口，内部保持清晰的职责边界。

#### B2.1 固定划分与确定性（保持原有存储结构）

- 目标
  - **保持原有的数据存储结构**，不引入新的目录层级或文件格式变化。
  - 严格遵循"一个版本的数据本体为单一文件（如 `data.pkl`）"，集合划分仅保存为索引文件。
  - loader 仅按索引加载，不在运行时生成/猜测。

- 方案
  - **维持现有存储结构**：
    ```
    data/<dataset>/
    ├── data.pkl           # 完整数据集（图结构 + 属性）
    ├── train_index.json   # 训练集索引列表
    ├── val_index.json     # 验证集索引列表
    └── test_index.json    # 测试集索引列表
    ```
  - **UnifiedDataInterface 适配**：
    - 内部方法 `_load_split_indices()` 直接从上述三个文件读取索引
    - 返回格式统一为 `{"train": [...], "val": [...], "test": [...]}`
    - 保留 `data_version` 和 `seed` 参数以便未来扩展，但当前忽略
  - **无需数据迁移**：
    - 现有数据文件结构不变，无需重新生成或移动文件
    - 所有现有的数据加载器继续正常工作

- 优点
  - 零迁移成本，完全向后兼容
  - 结构简单清晰，易于理解和维护
  - 专注于接口统一而非存储结构改变

#### B2.2 统一数据接口 UnifiedDataInterface（整合序列化/BPE，缓存为内部细节）

- 目标
  - 提供统一入口，屏蔽底层文件路径差异；支持“数据版本（data_version）发现与选择”；统一编排序列化与 BPE 两个高开销阶段，并将其缓存设计为接口内部可控的实现细节。
  - 默认严格只读：若缓存缺失则抛错；仅当显式要求构建时，才触发确定性构建。

- 接口设计（新文件 `src/data/unified_data_interface.py`）
  - 初始化
    - `UnifiedDataInterface(config: ProjectConfig, dataset: str)`
  - API（默认只读，禁止隐式写入）：
    - `get_graphs(data_version: str = "latest") -> List[GraphRecord]`
    - `get_sequences(method: str, data_version: str = "latest", *, build_if_missing: bool = False) -> Tuple[List[List[int]], Dict[str, List[int]]]`
    - `get_compressed_sequences(method: str, data_version: str = "latest", *, build_if_missing: bool = False) -> Tuple[List[List[int]], Dict[str, List[int]], Any]`
      - 返回 `(sequences_or_compressed, split_indices, bpe_model)`；其中 `split_indices` 来自规范 splits 文件。
      - 当 `build_if_missing=False`（默认）且缓存缺失时抛错；当 `True` 时，内部按确定性流程：
        1) 通过工厂/loader 读取 graphs 与属性；
        2) 执行序列化并持久化；
        3) 执行 BPE 训练与编码并持久化；
        4) 返回产物与 splits。
  - 版本发现/选择（澄清 version 语义）：
    - `data_version` 专指“图/属性数据版本”，可由 `data.pkl` 哈希或 `preprocessed_stats.json` 的版本字段确定；不混入 method 或 BPE 参数。
    - 序列化/BPE 的工艺参数由 `(dataset, method, bpe.num_merges, bpe.min_frequency, seed, ...)` 共同构成制品键（artifact key），用于组织缓存目录和文件名。
    - `latest` 对 `data_version` 的解析通过确定性排序（如最大时间戳或哈希序）；找不到则抛错（不做回退）。
  - 错误处理：
    - 任一关键文件（数据本体、序列化/BPE 结果、splits）缺失且 `build_if_missing=False` 时即抛 `FileNotFoundError`；不以默认路径或随机方式替代。

- 与现有 `quick_interface.py` 的关系
  - 过渡期：改为对 `UnifiedDataInterface` 的薄封装以维持兼容。
  - 重构完成后：删除 `quick_interface.py`，将调用点（含 BERT 流水线）统一迁移到 `UnifiedDataInterface`。

#### B2.3 工厂与职责边界

- 统一工厂（已具备）：`src/data/unified_data_factory.py` 继续作为数据集选择入口。
- 责任划分：
  - data 层：暴露 `get_node_token/get_edge_token/get_token_map/get_token_readable/get_node_type/get_edge_type` 等“数据集特定语义”。
  - serializer：使用 data 层接口，不自定义“化学类型/映射”的分支逻辑，不做任何猜测或回退。

---

### B3. 具体改造清单（按优先级，结合回退决策）

1) **保持原有数据存储结构**
- **不新增** `src/data/splits.py` 或 `scripts/generate_splits.py`
- 继续使用现有的 `data/<dataset>/` 下的三个索引文件
- UnifiedDataInterface 内部直接读取这些文件

2) 统一接口实现
- `src/data/unified_data_interface.py` 已实现 B2.2 所述 API 与显式构建策略（`build_if_missing`）。
- 内部方法 `_load_split_indices()` 读取原有的三个索引文件。
- `quick_interface.py` 已移至备份目录，调用点已迁移。

3) loader 收敛  
- BaseDataLoader 保持现有逻辑不变
- 错误文案已修正（移除 "MNIST" 相关硬编码）
  - 无文件时直接抛错，不回退。
- `QM9Loader/AQSOLoader/ZINCLoader/QM9TestLoader` 的 `_load_processed_data()` 保持“严格按索引切分”的逻辑不变。

4) 文档与用法
- 在 `DATA_GUIDE.md` 增补“数据目录结构与索引文件说明”，不引入新 splits 层级。

---

### B4. 测试与验收

- 单元测试
  - `tests/test_splits_determinism.py`
    - 同一数据规模与 seed，多次生成 splits 完全一致。
    - 不同 seed 生成的划分互斥性与比例检验（允许 ±1 误差）。
  - `tests/test_unified_data_interface.py`
    - `get_graphs/get_sequences/get_compressed_sequences` 正确返回并与底层缓存对齐；`latest` 选择确定性。
    - 当缓存缺失：`build_if_missing=False` 抛错；`True` 时能确定性构建、持久化，再次调用命中缓存。
  - `tests/test_loader_splits_path.py`
    - 修改后的 `get_split_indices()` 仅按 `ProjectConfig` 指定路径读取；缺失时抛错信息正确。

- 集成/端到端
  - `tests/test_e2e_pretrain_finetune_qm9test.py` 维持绿；当切换到 `UnifiedDataInterface` 时，训练数据获取逻辑等价且目录产物一致。

- 验收标准（对齐 TASK_PLAN）
  - 多次运行得到完全一致的数据划分；禁用内部随机切分。
  - 通过同一接口取 graphs/序列/BPE 压缩；切换数据版本无需改上游业务代码。
  - 修改或新增数据集时，无需改 serializer 接口。

---

### B5. 与配置/路径的对齐

- 配置读取
  - splits 目标路径、数据版本、seed 统一从 `ProjectConfig` 的 `dataset`/`paths` 段读取，禁止在子模块动态更改配置。

- 路径规范（建议）
  - `processed/<dataset>/preprocessed_data/`
  - `processed/<dataset>/serialized_data/<method>/`
  - `processed/<dataset>/bpe_compressed/<method>/`
  - （已回退）不引入 `processed/<dataset>/splits/<version>/seed{seed}.json`，保持 `data/<dataset>/` 三索引文件结构。

---

### B6. 迁移计划（无隐式回退，按照回退决策）

1) 保留 `data/<dataset>/` 三索引文件方案，无需生成新的 `splits/` 层级或脚本。
2) 引入并统一使用 `UnifiedDataInterface`；`quick_interface.py` 移至备份，调用点已迁移。
3) `BaseDataLoader.get_split_indices()` 与 `UnifiedDataInterface._load_split_indices()` 统一读取三索引文件，缺失即抛错。
4) 更新文档与示例，新增/更新测试直至绿。

---

### B7. 接口草案（示例）

```python
# src/data/unified_data_interface.py 内部方法
def _load_split_indices(self) -> Dict[str, List[int]]:
    """加载原有格式的划分索引文件（以配置的 data_dir 为基准）"""
    data_dir = Path(self.config.data_dir) / self.dataset
    splits = {}
    
    for split_name in ['train', 'val', 'test']:
        index_path = data_dir / f"{split_name}_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                splits[split_name] = json.load(f)
        else:
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
    
    return splits

# src/data/unified_data_interface.py
class UnifiedDataInterface:
    def __init__(self, config: ProjectConfig, dataset: str): ...
    def get_graphs(self, data_version: str = "latest") -> List[Dict[str, Any]]: ...
    def get_sequences(self, method: str, data_version: str = "latest", *, build_if_missing: bool = False) -> Tuple[List[List[int]], Dict[str, List[int]]]: ...
    def get_compressed_sequences(self, method: str, data_version: str = "latest", *, build_if_missing: bool = False) -> Tuple[List[List[int]], Dict[str, List[int]], Any]: ...
```

---

### B8. 对现有代码需修复的小问题

- ✅ `BaseDataLoader.get_split_indices()` 的异常信息已修正为通用表述。
- 明确在注释与文档中声明：`TRAIN_RATIO/VAL_RATIO/TEST_RATIO` 只用于元数据展示，不参与运行时切分。
- **重要决定**：保持原有数据存储结构（`data/<dataset>/` 下的三个索引文件），避免不必要的迁移成本。

---

### B9. 预期影响

- 上游（序列化/BPE/训练）调用更稳定，路径/版本切换统一可控。
- 数据划分行为完全可复现、可审计；新增数据集或替换数据版本的改动对上层透明。
- 与 `A` 任务中统一化的实验路径/命名/快照策略契合，便于日志与结果对照分析。

---

### B10. 里程碑与产物

- 代码改动：
  - ✅ `src/data/unified_data_interface.py` - 新增统一数据接口
  - ✅ 修改 BERT 训练流程使用 UDI
  - ✅ 移除 `quick_interface.py` 依赖
  - ✅ 修复 `BaseDataLoader.get_split_indices()` 错误信息
- 文档：
  - ✅ 当前文件 `docs/TASKS_B_DATA.md` 更新
  - `DATA_GUIDE.md` 待更新
- 测试：新增/更新单测与端到端用例。


