## 序列化与数据层张量化重构执行计划（冻结，待实施）

本文件固化当前针对序列化与数据层张量化重构的详细执行计划，用于在 Pipeline 设计阶段完成后作为实施蓝图。当前阶段仅记录规划，不进行代码改动。

### 说明与当前状态
- 项目正处于 Pipeline 重构与设计阶段；为避免在不稳定期引入额外变量，以下改动暂不实施。
- 规划依据：`docs/DESIGN_tensor_refactor.md`、`docs/TASKS_C_SERIALIZER.md`、`docs/PERF_PLAN.md`、`docs/TASK_PLAN.md` 及现有源码剖析。
- 原则：单一配置源、无隐式回退、使用真实数据、可复现；优先保证正确性与确定性，其次优化性能。

### 总目标
- 全面张量化装配与查询，消除逐项 Python 调用与不必要的 NetworkX 依赖。
- 序列化算法（Euler/Feuler/CPP/FCPP）在保证输出一致性的前提下获得可观加速。
- 统一接口、统一错误行为，跨数据集确定性一致。

### 范围与非目标
- 范围：`src/algorithms/serializer/*` 与 `src/data/*` 的交互层、装配与邻接/连通访问、最短路实现替换、性能基准与测试。
- 非目标：不在此阶段调整训练/微调模型结构；不产出“结论性报告”。

### 优先级任务清单

- P0｜序列化算法加速（核心）
  - 改动要点
    - 去除热路径 `convert_dgl_to_networkx` 依赖，邻接/度/连通改用张量/CSR。
    - Feuler 寻路阶段：优先级比较与三元组频率查表向量化；消除列表操作与重型 Python 控制流。
    - 单路径 `edge_ids` 仅一次批量计算并复用；从热路径移除 `_build_edge_id_mapping` 依赖与使用。
    - Feuler tie-breaker：同权邻居按邻居 ID 升序，确保确定性。
  - 触点文件
    - `src/algorithms/serializer/freq_eulerian_serializer.py`
    - `src/algorithms/serializer/eulerian_serializer.py`
    - `src/algorithms/serializer/base_serializer.py`（移除 `_build_edge_id_mapping` 使用；保留批量 `edge_ids` 流程）
  - 验收标准
    - `qm9test` 全量：序列化（寻路+装配）总耗时较当前基线再降 ≥20%，且与缓存对齐 mismatch=0。
    - 热路径中不再出现 `convert_dgl_to_networkx`；单路径最多一次 `edge_ids` 计算。

- P0｜连通性/邻接张量化替换 NetworkX（热路径）
  - 改动要点
    - 提供基于 `src/dst` 的 CSR/张量 BFS/并查集工具，替代 Python 邻接构建与 `to_networkx`。
  - 触点文件
    - `src/algorithms/serializer/utils.py`（或新增 `tensor_graph_utils.py`）
    - 调用方：`eulerian`/`feuler` 内部邻接访问处
  - 验收标准
    - profiler 中 `convert_dgl_to_networkx` 占比≈0；功能一致，mismatch=0。

- P1｜接口一致性与确定性加固
  - 改动要点
    - `BaseGraphSerializer.multiple_serialize` 缺 `dgl_graph` 时与 `serialize` 一致抛 `ValueError`；
    - `tokens_to_string` 的断言与报错信息一致；
    - `eulerian_serializer.py` 使用邻接前排序，避免依赖 DGL 内部边序；
    - Feuler 源码与头注释明确“仅使用三元组频率；两跳统计暂不启用（保留注释）”。
  - 触点文件
    - `src/algorithms/serializer/base_serializer.py`
    - `src/algorithms/serializer/freq_eulerian_serializer.py`
    - `src/algorithms/serializer/eulerian_serializer.py`
  - 验收标准
    - 四个入口方法错误行为一致；同一图多次运行输出一致。

- P1｜数据层抽象清理与批量接口张量化核查
  - 改动要点
    - `BaseDataLoader`：仅保留抽象的 `get_most_frequent_edge_type`，移除重复空实现，避免静默返回。
    - 核查/补全批量接口：`get_node_tokens_bulk`/`get_edge_tokens_bulk`/`get_node_types_bulk`/`get_edge_types_bulk` 均为张量化实现；`get_graph_src_dst`、`ndata`/`edata` 字段完备。
  - 触点文件
    - `src/data/base_loader.py`、`src/data/*_loader.py`
  - 验收标准
    - 序列化层不出现逐项 get_* 循环；相关单测通过。

- P1｜避免 DGL 图频繁重建
  - 改动要点
    - `topo`/`bfs`/`dfs` 等路径：以张量筛选/视图式索引替代 DGLGraph 重建与特征拷贝。
  - 触点文件
    - `src/algorithms/serializer/topo_serializer.py`、`bfs_serializer.py`、`dfs_serializer.py`
  - 验收标准
    - 构图调用次数较当前下降 ≥90%；功能一致。

- P2｜CPP/FCPP 路径完善
  - 改动要点
    - 确认并完成 SciPy `csgraph.shortest_path` 在 FCPP 的最短路替换；匹配保留 NetworkX（Blossom）。
    - 评估匹配阶段替代/降维策略（提供可选开关）。
  - 触点文件
    - `src/algorithms/serializer/freq_chinese_postman_serializer.py`
    - `src/algorithms/serializer/chinese_postman_serializer.py`
  - 验收标准
    - `nx_dijkstra` 占比显著下降；功能一致；匹配优化作为可选开关且 mismatch=0。

- P0｜测试与基准（与实现同步推进）
  - 改动要点
    - 增补“序列化等价性对齐测试”：对比 processed 缓存逐样本 mismatch=0；
    - 增补性能基准脚本/用例：输出模块占比，验证“去 NetworkX/批量 edge_ids/向量化”收益。
  - 触点
    - `tests/8.9_refactor/taskC/*`、`docs/PERF_PLAN.md`、基准脚本
  - 验收标准
    - 本地可复现实验；基准输出结构化指标；串行/并行等价测试通过。

### 基线与验证数据
- 数据集：`qm9test` 全量（固定随机种子与依赖版本）。
- 基线：见 `docs/PERF_PLAN.md` 的当前耗时概览与分解。
- 成果要求：修改前后对齐测试 mismatch=0；性能指标满足各任务验收标准。

### 依赖与前置
- Pipeline 重构完成并稳定；
- 统一配置入口与命名、日志结构既已落实；
- 真实数据与 processed 缓存可读取。

### 风险与约束
- 遵循“无隐式回退”：失败立即抛错并给出指示，避免掩盖问题；
- 去 NetworkX 与向量化可能引入行为差异：先做等价性测试再推广；
- 并行仅作为可选优化，默认关闭，确保功能一致。

### 执行顺序建议（不含时间指标）
1) P0：序列化算法加速 + 去 NetworkX（euler/feuler）→ 跑 `qm9test` 等价性与性能基线；
2) P1：接口一致性/确定性 + 数据层抽象清理与批量接口核查；
3) P1：避免 DGL 重建（topo/bfs/dfs）；
4) P2：FCPP 最短路替换完善与匹配可选优化路径。

### 变更记录
- 2025-08-09：创建冻结执行计划，暂不实施，待 Pipeline 稳定后按此计划推进。






