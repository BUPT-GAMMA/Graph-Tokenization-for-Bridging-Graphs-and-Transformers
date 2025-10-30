# 数据层说明（已对齐 8.9 重构）

## 核心结论

- 固定划分：统一采用 `data.pkl + train_index.json/val_index.json/test_index.json`，缺失即报错，不做隐式回退。
- 统一接口：通过 `UnifiedDataInterface` 提供图、序列化结果、BPE 压缩结果的统一读取与（显式）构建。
- 工厂与职责：`UnifiedDataFactory` 选择数据集 loader；token 规则由数据层暴露，序列化器只读取不介入。
- **严格数据契约**：预处理与加载器之间建立严格的数据格式契约，不做兼容性假设（详见 CODING_STANDARDS.md 第5章）。

## 目录结构（现状）

```
src/data/
├── DATA_LAYER_NEW.md             # 本文档（当前）
├── README.md                     # 历史文档（保留，顶部有迁移提示）
├── base_loader.py                # 抽象基类：统一索引读取/统计/校验
├── unified_data_factory.py       # 工厂：注册与创建各数据集加载器
├── unified_data_interface.py     # 统一数据接口：图/序列/BPE 读取与显式构建
├── qm9_loader.py                 # QM9 加载器
├── qm9test_loader.py             # QM9Test 加载器（10%）
├── zinc_loader.py                # ZINC 加载器
├── aqsol_loader.py               # AQSOL 加载器
├── mnist_loader.py               # MNIST 超像素图加载器
└── single_graph_loader.py        # 单图采样工具（研究用途）
```

## 固定划分与确定性

- 索引读取：`base_loader.py:get_split_indices()` 统一读取三份索引 JSON，异常直接抛出。
- 各 loader 均按索引切分数据，不做随机划分：
  - `qm9_loader.py`、`qm9test_loader.py`、`zinc_loader.py`、`aqsol_loader.py`、`mnist_loader.py`
- 数据文件：各数据集严格检查 `data.pkl` 是否存在；不存在即抛 `FileNotFoundError`。

## 统一数据接口 UnifiedDataInterface

统一入口：`src/data/unified_data_interface.py`

- 图数据
  - `get_graphs()`：返回全量图（不做切分）。
  - `get_graphs_by_split(split)`：按索引切分后的子集。
- 核心序列化接口（简化版）
  - `get_sequences(method)`：获取所有序列数据和标签，返回 `([(图ID, 序列), ...], [属性字典, ...])`。
  - `get_sequences_by_splits(method)`：按训练/验证/测试划分获取序列数据，返回六元组 `(train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels)`。
  - `get_training_data(method, target_property=None)`：便捷训练接口，返回纯序列和提取的标签。支持自动获取target_property：单属性数据集自动选择，多属性数据集使用默认属性或要求明确指定。
- 辅助接口
  - `get_vocab(method)`：读取与数据集绑定的完整词表（包含原始token + BPE合并token + 特殊token），缺失直接报错。
  - `get_split_indices()`：获取数据集划分索引。
  - `get_downstream_metadata()`：返回下游任务相关的元信息。
  - `get_bpe_codebook(method)`：读取 BPE codebook。

数据管理：简化为单一数据版本，序列化结果缓存在 `processed_data_dir/<dataset>/serialized_data/<method>/` 下。

## 数据工厂与职责边界

- 工厂：`unified_data_factory.py` 统一注册/创建 loader。
- Token 职责：数据层实现 `get_node_token/get_edge_token/get_token_map/get_*_type`；序列化器仅调用数据层获取 token。
  - 对应调用点：`src/algorithms/serializer/base_serializer.py` 中 `get_node_token/get_edge_token`。

## 最简用法示例

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface

cfg = ProjectConfig()
cfg.dataset.name = "qm9test"

udi = UnifiedDataInterface(cfg, cfg.dataset.name)

# 读取图
graphs = udi.get_graphs()

# 读取序列化结果（所有数据）
sequences, labels = udi.get_sequences(method="feuler")

# 读取按划分的训练数据（带图ID）
tr_seqs, tr_labels, va_seqs, va_labels, te_seqs, te_labels = udi.get_sequences_by_splits(method="feuler")

# 便捷训练接口（纯序列 + 提取标签）
# 自动获取：对于单属性数据集（如ZINC、AQSOL），无需指定target_property
tr_sequences, va_sequences, te_sequences, tr_y, va_y, te_y = udi.get_training_data(method="feuler")

# 明确指定：对于多属性数据集（如QM9），建议明确指定
tr_sequences, va_sequences, te_sequences, tr_y, va_y, te_y = udi.get_training_data(method="feuler", target_property="homo")

# 读取词表
vocab_manager = udi.get_vocab(method="feuler")

# 读取划分索引
split_indices = udi.get_split_indices()
```

## 规则与错误策略

- 缺失文件直接报错，不进行静默回退；数据构建通过独立的 `data_prepare.py` 脚本完成。
- 序列化结果与词表必须预先构建，UDI 仅提供读取接口。
- 所有路径解析均基于 `ProjectConfig` 的绝对路径，避免依赖当前工作目录。

## 后续计划（版本化）

- `data_version` 将支持：
  - 目录结构：`processed_data_dir/<dataset>/<version>/...`
  - 发现 API：`list_available_versions()` 与 `resolve_version("latest")`
  - 测试：指定版本/最新版本读取与构建的对齐用例

## 测试覆盖

- 接口与错误：`tests/8.9_refactor/taskB/test_unified_data_interface.py`、`test_unified_data_interface_errors.py`、`test_udi_new_methods.py`
- 数据集一致性：`tests/8.9_refactor/taskB/test_dataset_equivalence_qm9_vs_qm9test.py`
- 端到端：`tests/8.9_refactor/test_e2e_pretrain_finetune_qm9test.py`

