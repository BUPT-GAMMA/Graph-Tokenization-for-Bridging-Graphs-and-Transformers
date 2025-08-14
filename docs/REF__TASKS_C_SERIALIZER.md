## C. 序列化算法层重构规划（Part C）

本文件基于现有实现与《docs/TASK_PLAN.md》中 C1–C3 的目标，给出：现况、必须改动、目标效果与验收标准。范围涵盖 `src/algorithms/serializer/` 以及与数据层的交互。

---

### C1. 接口与命名统一（命名调整：graph_seq → feuler，eulerian 保持 eulerian）

- 目标
  - 统一四个入口与签名，并返回标准 `SerializationResult`：
    - `serialize(graph)`
    - `multiple_serialize(graph, num_samples: int)`
    - `batch_serialize(graphs)`
    - `batch_multiple_serialize(graphs, num_samples: int)`
  - 任意序列化器都能以上述统一方式被上层调用。

- 现况
  - `src/algorithms/serializer/base_serializer.py` 已提供统一接口与 `SerializationResult`，各子类通过 `_serialize_single_graph` 对接。
  - 工厂 `serializer_factory.py` 已统一创建并透传必要参数；方法名已统一为 `feuler` 与 `eulerian`。
  - 实现文件：
    - Feuler 方法对应实现位于 `src/algorithms/serializer/freq_eulerian_serializer.py`（类名 `FeulerSerializer`）。
    - Eulerian 方法对应实现位于 `src/algorithms/serializer/eulerian_serializer.py`（类名 `EulerianSerializer`）。
  - 小问题：
    - `BaseGraphSerializer.multiple_serialize` 在缺少 `dgl_graph` 时返回空结果，而 `serialize` 会抛错，行为不一致。
    - 文案提示不一致：`tokens_to_string` 对 `get_token_readable` 的断言与报错信息不吻合。

- 必须改动（不引入任何“回退”逻辑）
  - 在 `BaseGraphSerializer`：
    - 统一错误处理：`multiple_serialize` 缺 `dgl_graph` 时与 `serialize` 一致抛 `ValueError`，不返回空序列。
    - 文案修正：`tokens_to_string` 对 `get_token_readable` 的断言与提示一致。
  - 文档同步：方法名统一为 `feuler` 与 `eulerian`；示例与清单不再出现 `graph_seq` 或 `euler` 简写。

- 验收标准
  - 四个入口方法在所有序列化器上均可调用、返回 `SerializationResult`，且错误场景行为一致（统一抛错）。

---

### C2. 确定性与多数据集兼容（含 Feuler 统计项说明）

- 目标
  - 同一图在同一配置下，输出完全一致；
  - 不依赖隐式数据来源，数据集特定信息（如 token 映射、常见边类型）从 data 层显式获取；
  - 跨数据集不出现未定义分支或静默回退。

- 现况
  - `BFS/DFS`：邻居访问基于排序，确定性良好。
  - Feuler（位于 `freq_eulerian_serializer.py`）：初始化收集三元组频率，路径选择按权重降序；
    - 风险点：权重相等时未明确二级排序（tie-breaker），可能在不同运行环境出现顺序差异；
    - 关于 `_extract_all_statistics`：当前实现“仅统计三元组频率”，两跳路径统计的两条路径（注释掉的部分）“暂不纳入统计与决策”，这是**预期内的行为**；
      - 要求：保留该段注释代码，并在文档与代码注释中明确“出于研究设计，暂不启用两跳路径统计”，以免造成误解。
  - `EulerianSerializer`：邻接基于 DGL 的边顺序，为跨平台稳定性建议在使用前排序；
  - `BaseDataLoader`：`get_most_frequent_edge_type` 被重复定义（先 abstract，后空实现覆盖），风格不一致，应仅保留抽象方法；
  - 数据层：`QM9Loader/QM9TestLoader` 已实现 `get_node/edge_token` 与 `get_most_frequent_edge_type` 等接口，序列化侧无需读取原始特征细节。

- 必须改动
  - 在 Feuler（`freq_eulerian_serializer.py`）：
    - 在选择下一边时，对同权重邻居追加二级排序（邻居 ID 升序）确保稳定；
    - 在 `_extract_all_statistics` 与模块头注释增加说明：当前“仅使用三元组频率”；两跳路径统计代码保留为注释，注明“暂不启用，预期内”。
  - 在 `eulerian_serializer.py`：
    - 使用/构建邻接列表后对邻居进行排序，避免依赖 DGL 内部边序。
  - 在 `base_loader.py`：
    - 移除重复的空实现，只保留抽象的 `get_most_frequent_edge_type`，防止子类遗漏实现时“静默返回 None”。

- 验收标准
  - 对固定输入图，多次运行输出 token 序列完全一致；
  - `qm9test` 与另一数据集的最小样例上可直接运行，无 KeyError/未实现分支；
  - 代码中无通过“返回空值/默认值”掩盖问题的逻辑。

---

### C3. 批处理并行（暂缓，待 C1/C2 完成并充分测试后再实施）

- 决策
  - 先聚焦 C1/C2 的命名统一与确定性加固，并完成端到端与单元测试全绿；
  - C3 并行化作为后续性能优化项，单独开一个阶段推进，避免与命名/接口重构耦合带来排查困难。

- 规划（供后续参考，非本阶段任务）
  - 在 `BaseGraphSerializer` 为批量接口增加可选并行参数（默认关闭），严格保序；
  - 提供独立基准脚本，验证串行/并行等价性与性能收益；
  - 文档与测试在进入 C3 时同步补充。

---

### 代码触点清单（按文件）

- `src/algorithms/serializer/base_serializer.py`
  - 统一错误处理（`multiple_serialize` 缺 `dgl_graph` → 抛 `ValueError`）；
  - `tokens_to_string` 的断言提示修正；
  - 为批量接口添加并行可选参数，确保保序；
  -（可选）为邻接/边 ID 查询等工具函数补充“排序/稳定性”说明。

- `src/algorithms/serializer/freq_eulerian_serializer.py`
  - tie-breaker：同权邻居按 ID 升序；
  - 清理 `_extract_all_statistics` 的死代码；
  - 在 `_dataset_stats` 中明确当前使用的统计项。

- `src/algorithms/serializer/eulerian_serializer.py`
  - 邻接使用前排序，保证不同运行环境下的确定性。

- `src/algorithms/serializer/topo_serializer.py`
  - 更正错误导入。

  - `src/algorithms/serializer/README.md`
    - 术语与方法名统一为 `feuler`、`eulerian`、`topo`；
    - Feuler 对应文件为 `freq_eulerian_serializer.py`，Eulerian 对应 `eulerian_serializer.py`，Topo 对应 `topo_serializer.py`；
    - 更新统一接口片段与并行开关说明（默认串行，显式启用并行）。

- `src/data/base_loader.py`
  - 移除 `get_most_frequent_edge_type` 重复空实现，仅保留抽象方法。

---

### 迁移与兼容

- 对上层调用方：保持方法名与返回结构不变（已有统一接口），不需要改动业务逻辑；
- 对配置：并行为显式参数（默认关闭），不开启时行为完全一致；
- 对文档：本文件与 `serializer/README.md` 同步更新，确保“文件名/接口/示例”一致。

---

### 建议测试（新增/强化）

  - 接口统一
    - 针对每个序列化器，调用四个入口，校验 `SerializationResult` 可读可取；缺 `dgl_graph` 统一抛错。

  - 确定性
    - 固定种子/固定输入图，多次运行 `feuler`/`eulerian`/`bfs` 输出完全一致；构造同权边用例验证 tie-breaker 生效。

  - 跨数据集
    - 在 `qm9test` 与另一个数据集的小样上运行 `feuler`/`bfs`，无 KeyError/未实现分支。

- 并行一致性与基准
  - 串行 vs 并行 逐项完全相等；打印耗时对比报告，供人决定是否开启并行。

---

### 结论

本次 Part C 重构以“接口统一、确定性优先、并行显式开启且保序”为核心原则。改动集中在少量触点，不影响上层业务调用；完成后将保证跨数据集/跨环境的稳定输出，并在需要时可获得可重复的并行加速。


