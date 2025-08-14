## 图序列化层张量化重构设计

### 目标
- 抛弃 for-loop/逐项 Python 调用的数据准备/装配，端到端使用张量操作；类型用整数 ID；接口直接返回整图张量。
- element_sequences 与 token 序列等价处理：均作为 LongTensor 统一装配；仅在需要可读化输出时再做映射。
- 不做回退；错误立即抛错；优先性能。

### 数据层（Loader v2）
新增整图张量接口（必须实现）：
- `get_graph_node_type_ids(g) -> LongTensor[N]`
- `get_graph_edge_type_ids(g) -> LongTensor[E]`
- `get_graph_node_token_ids(g) -> LongTensor[N, Dn]`
- `get_graph_edge_token_ids(g) -> LongTensor[E, De]`
- `get_graph_src_dst(g) -> (LongTensor[E], LongTensor[E])`

构建时预写入图特征：
- `g.ndata['node_type_id']`, `g.edata['edge_type_id']`
- `g.ndata['node_token_ids']`, `g.edata['edge_token_ids']`

说明：bulk 接口保留，但实现基于张量索引；严禁回退到逐项。

### 序列化层改造
- `serialize()`：一次性取整图张量（类型、token、src/dst），后续不再调用逐项 get_*。
- 路径→tokens：
  - 节点 token：`node_token_ids.index_select(0, node_path)` → `[P, Dn]`
  - 边 token：`edge_ids = g.edge_ids(node_path[:-1], node_path[1:])` → `edge_token_ids[edge_ids]`
  - element_sequences：与 token 同步按张量装配（例如节点位形成 `[P, 1]` 的元素标签、边位形成 `[P-1, 1]` 的元素标签，或拼为单一平铺序列 `[L]` 与对应 segment 标注），在需要人类可读时再映射为字符串。
  - 最终输出尽可能保持为 LongTensor（必要时再转 list）。
- 统计/权重：由 `src/dst`、`node_type_ids`、`edge_type_ids` 一次性生成三元组并查权重（优先 3D 张量；备用稀疏哈希表）。
- 邻接/度/连通：优先 CSR/张量操作，避免 `to_networkx`。

### 算法侧特殊说明
- Euler/Feuler：主循环保留最小 Python 控制流；邻接预排序用张量 + 指针索引；禁止 list.pop(0)。
- CPP/FCPP：多源最短路改 SciPy 稀疏最短路；匹配先保留 networkx；其余用张量化取数。

### 触点清单
- `src/data/base_loader.py`：整图张量接口（抽象，禁止回退）与 bulk 的张量实现约束
- `src/data/qm9_loader.py`：加载后一次性写入四类特征并实现整图接口
- `src/algorithms/serializer/base_serializer.py`：张量化 `convert_path_to_tokens`（保留 list 版仅用于极端回溯调试）
- `eulerian_serializer.py`、`freq_eulerian_serializer.py`：邻接/权重用张量；主循环轻量
- `chinese_postman_serializer.py`、`freq_chinese_postman_serializer.py`：SciPy 最短路

### 里程碑与验证
- Phase A：数据层整图接口 + 上层张量取数；剖析 euler/feuler 关键时间下降
- Phase B：CPP/FCPP 最短路换 SciPy；`nx_dijkstra` 明显下降
- Phase C：Euler/Feuler 邻接张量化；`feuler_find` 占比下降

### 开放决定点（需要与用户确认时）
- 三元组频率查表若使用 3D 张量，会占用 `num_node_types * num_edge_types * num_node_types` 的内存，是否接受？
  ——已确认，采用高速方案
 - 三元组频率查表我们将采用 3D 张量（类型空间较小、可接受内存开销）；如后续跨数据集类型空间显著增大，再评估稀疏方案。
  ——已确认，采用张量方案



### 进度与状态（2025-08-09）

- 已完成
  - 数据层（Loader v2）
    - 在 `src/data/base_loader.py` 落实整图张量接口抽象：`get_graph_node_type_ids`/`get_graph_edge_type_ids`/`get_graph_node_token_ids`/`get_graph_edge_token_ids`/`get_graph_src_dst`（禁止回退）。
    - 在 `src/data/qm9_loader.py` 加载阶段一次性写入整图特征：
      - 节点：`ndata['node_type_id']`、`ndata['node_token_ids']`
      - 边：`edata['edge_type_id']`、`edata['edge_token_ids']`
    - 批量 API 基于张量实现（`get_*_tokens_bulk` / `get_*_types_bulk`）。
  - 序列化层基础张量化装配
    - `BaseGraphSerializer._convert_path_to_tokens` 改为全张量流程：节点 `index_select` 一次 gather；边 `g.edge_ids(node_ids[:-1], node_ids[1:])` 一次计算并复用；
    - 最高频边省略：基于 `edge_type_ids` 构造掩码一次性过滤；
    - 输出顺序：恢复“交错（node_i 紧跟 edge(i,i+1)）”为标准；
    - element 暂不使用，返回等长占位（仅 tokens 保持 LongTensor→末端转 list）。
  - 统计/权重张量化
    - 在基类统计收集阶段构建 3D 三元组频率张量 `freq[v, e, v]` 与其归一化版本（>0 位置 min-max）。
    - `BaseGraphSerializer._calculate_edge_weights` 支持张量化查表：`log10(freq[src_t, et_t, dst_t])`。
    - `FeulerSerializer` 使用上述权重；`FCPPSerializer` 用归一化频率并将最短路替换为 SciPy `csgraph.shortest_path`（CSR），匹配仍保留 NetworkX（Blossom）。
  - 并行化与一致性
    - 基类 `batch_serialize`/`multiple_serialize` 线程池并行（分片/线程本地状态），并行与串行结果严格一致（测试已通过）。
    - 交错顺序回退后，与历史缓存（`aqsol`/`mnist`）保持一致；`qm9/qm9test/zinc` 历史分块结果误覆盖已回溯，现以交错为标准并备份参考集（`data/processed_backups/…`）。
  - 兼容性与确定性
    - 建立“单点修改→对照 processed 缓存逐样本比对（mismatch=0）→再推进”的工作流；
    - 禁止回退；错误直接抛出并附带指示信息；统一内部 LongTensor 表示。

- 未完成/进行中
  - 序列化算法加速（核心）【未完成】
    - 目标：Euler/Feuler 邻接张量化；寻路主循环进一步消除 Python 控制流/容器；优先级比较与三元组频率查表在寻路阶段的向量化；确保单路径 `edge_ids` 仅一次批量获取与复用。
  - 连通性/邻接【未完成】
    - 用 CSR/张量 BFS/并查集替换 `convert_dgl_to_networkx` 与 Python 邻接构建。
  - 移除热路径上的 `_build_edge_id_mapping`【未完成】
    - 统一批量 `edge_ids`，禁止逐步查询与重复计算。
  - `topo`/`bfs`/`dfs` 相关【未完成】
    - 避免频繁重建 DGL 图，改为张量筛选/视图式索引。
  - 匹配阶段（Blossom）优化/替代【后续阶段】
    - 属于工程复杂项，保留为后续探索。

- 代表性性能（qm9test，全量，阶段性结果）
  - Eulerian：total ≈ 20.6s → 15.6s；`convert_path_to_tokens` ≈ 12.8s → 7.2s。
  - Feuler：权重与寻路相关耗时下降；
  - CPP/FCPP：装配下降；最短路换 SciPy 后占比下降但匹配仍为瓶颈。
  - 说明：上述收益主要来自“装配层张量化”；“序列化算法加速（核心）”尚未完成，目标是在 qm9test 上总耗时在此基础上再下降 ≥20%（以当前基线为准），且功能一致（mismatch=0）。

- 近期执行计划（按优先级与验收标准）
  - P0｜序列化算法加速（核心）
    - 内容：Euler/Feuler 邻接张量化；寻路 inner-loop 去 Python 控制流；三元组频率查表在寻路阶段向量化；单路径 `edge_ids` 一次批量获取与复用。
    - 验收：qm9test 上与当前基线相比，序列化相关总耗时（寻路+装配）再下降 ≥20%；缓存对齐比对 mismatch=0。
  - P0｜连通性/邻接张量化替换 NetworkX
    - 内容：移除热路径上的 `convert_dgl_to_networkx` 与 Python 邻接构建，采用 CSR/张量 BFS/并查集。
    - 验收：热路径不再出现 `convert_dgl_to_networkx` 调用；profile 占比≈0；功能一致。
  - P1｜移除 `_build_edge_id_mapping`，统一批量 `edge_ids`
    - 内容：从所有热路径移除该映射；仅保留一次性批量 `edge_ids`。
    - 验收：单路径最多一次 `edge_ids` 计算并复用；无逐步查询。
  - P1｜避免 DGL 图频繁重建（`topo`/`bfs`/`dfs`）
    - 内容：以张量筛选/视图式索引替代重建。
    - 验收：构图调用次数较当前下降 ≥90%；profile 证实热路径不再重建。
  - P2｜FCPP 匹配阶段优化/替代探索
    - 内容：在保持确定性与一致性的前提下，评估替代算法或降维策略，提供可选开关。
    - 验收：开启优化不改变功能输出（mismatch=0），并具有可观性能收益或更低内存占用。

- 验证与基线
  - 数据集：`qm9test` 全量
  - 度量：总耗时、模块占比（profile）；功能一致性（mismatch=0 与已备份 processed 缓存）
  - 控制：固定随机种子与依赖版本
