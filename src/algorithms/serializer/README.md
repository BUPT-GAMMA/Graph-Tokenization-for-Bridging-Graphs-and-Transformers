# 图序列化算法架构文档（对齐当前实现）

## 概述

本模块将图结构转换为可供序列模型使用的 token 序列。所有序列化器遵循统一接口，并由工厂统一创建与注册。

## 组件与文件

- 基类与统一接口：`base_serializer.py`
- 工厂：`serializer_factory.py`
- 具体实现：
  - 频率引导欧拉（原 graph_seq，对外名 `feuler`）：`freq_eulerian_serializer.py`
  - 欧拉回路：`eulerian_serializer.py`
  - BFS：`bfs_serializer.py`
  - DFS：`dfs_serializer.py`
  - 拓扑排序：`topo_serializer.py`
  - SMILES：`smiles_serializer.py`（同时注册 `smiles_1`〜`smiles_4`）
  - 中国邮路（`cpp`）：`chinese_postman_serializer.py`
  - 频率引导邮路（`fcpp`）：`freq_chinese_postman_serializer.py`

工厂可用方法（与当前代码一致）：

```python
from src.algorithms.serializer.serializer_factory import SerializerFactory
SerializerFactory.get_available_serializers()
# → ['smiles_1','smiles_2','smiles_3','smiles_4','dfs','bfs','eulerian','topo','feuler','cpp','fcpp']
```

## 使用方式（推荐）

结合统一数据接口 `UnifiedDataInterface`：

```python
from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory

cfg = ProjectConfig()
cfg.dataset.name = 'qm9test'
udi = UnifiedDataInterface(cfg, cfg.dataset.name)
loader = udi.get_dataset_loader()
graphs = udi.get_graphs()  # 若使用 feuler，建议传入少量图用于统计

serializer = SerializerFactory.create_serializer('feuler')
serializer.initialize_with_dataset(loader, graphs[:20])

res = serializer.serialize(graphs[0])
token_seq, element_seq = res.get_sequence(0)
```

批量与多次序列化（统一返回 `SerializationResult` 或其列表）：

```python
serializer.multiple_serialize(graphs[0], num_samples=3)
serializer.batch_serialize(graphs[:16])
serializer.batch_multiple_serialize(graphs[:16], num_samples=2)
```

错误行为统一：缺少 `dgl_graph` 或图为空时抛 `ValueError`。

## 重要实现细节（确定性与参数）

- Feuler（`freq_eulerian_serializer.py`）
  - 仅使用“三元组频率”作为引导；两跳统计“预期不启用”（保留注释说明）。
  - 同权重邻居使用二级排序（邻居 ID 升序）保证确定性。
  - 初始化需传入部分图以收集统计信息。
  - 参数：`include_edge_tokens`、`omit_most_frequent_edge`。

- Eulerian（`eulerian_serializer.py`）
  - 使用前对邻接列表的邻居排序，避免依赖 DGL 内部边序，保证跨环境稳定。

- Topo（`topo_serializer.py`）
  - 基于拓扑排序（会为无向图定向），队列取出前按索引排序以保证确定性。

## 返回类型

序列化结果为 `SerializationResult`：

```python
res = serializer.serialize(graph)
token_seq, element_seq = res.get_sequence(0)
# element_seq 中的元素形如：'node_{i}', 或分段标记 'START_NODE_i'/'END_NODE_i'，以及可选边元素标记
```

说明：数据落盘与属性拼接的字典结构由数据接口 `UnifiedDataInterface` 负责，非序列化器直接职责。

## 与配置/路径的约定

方法目录名由配置计算：`{serialization.method}-{BPE|RAW}`（例如 `feuler-BPE`）。默认方法为 `feuler`（见 `config/default_config.yml`）。

## 常见用法片段

```python
# 包含边且省略最高频边类型（更短序列）
ser = SerializerFactory.create_serializer('feuler', include_edge_tokens=True, omit_most_frequent_edge=True)

# 仅节点序列
ser = SerializerFactory.create_serializer('feuler', include_edge_tokens=False)
```

## 兼容与边界

- 所有序列化器均实现统一四入口：`serialize` / `multiple_serialize` / `batch_serialize` / `batch_multiple_serialize`。
- 缺关键字段或不满足前置条件时应显式抛错；不使用静默回退。

## 备注（与历史命名的对应）

- 过去的 `graph_seq` 现统一为方法名 `feuler`；实现文件为 `freq_eulerian_serializer.py`。
- 文档与示例中的 `topological` 已统一为方法名 `topo`，文件为 `topo_serializer.py`。
