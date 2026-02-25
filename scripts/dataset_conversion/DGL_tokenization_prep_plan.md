## DGL 图级数据集 Token 化预处理方案（节点/边）

本文档汇总我们对若干 DGL/TU 图级任务数据集的字段核验结果，并给出统一的 Token 化预处理规范：对节点侧与边侧，如何从现有字段中生成离散的 `token` 数值（整数）。文末附统一的存储约定与实现要点，便于后续直接落地实现。

参考链接：
- TU 数据集总表（含 Node/Edge Labels/Attr. 标注）: `https://chrsmrrs.github.io/datasets/docs/datasets/`
- DGL 图级任务类型数据集列表: `https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.data.html#graph-prediction-datasets`
- DGL `TUDataset` 文档: `https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset`


## 统一预处理规范（结论）

- 节点侧（node token）
  - PROTEINS、COLORS-3（将多/one-hot 转为离散数值）、COIL-DEL（两个维度相乘后取整数）: 使用节点 `attr` 作为 token 源。
  - SYNTHETIC、Mutagenicity、DBLP_v1、DD、TWITTER-Real-Graph-Partial: 使用节点 `label` 作为 token 源；先转换成每节点一个整数的 `node_attr`，再以该 `node_attr` 作为 token 值。

- 边侧（edge token）
  - 若同时无 `edge_attr` 与 `edge_labels`，则以常数 0 作为边 token 值（为每条边写入 0）。
  - 否则一律使用 `edge_labels`（或 `label`）作为 token 源；若为 one-hot/多热/其它结构，先转换为单一整数，再保存为 `edge_attr`，作为 token 值。

- 统一存储约定
  - 生成字段名：节点侧 `g.ndata['token']`，边侧 `g.edata['token']`（均为整型张量）。
  - 保留原字段：不覆盖 `node_labels` / `node_attr` / `edge_labels` / `edge_attr` 等原有键。
  - 可选辅助：如需追踪映射，可保存 `token_mapping_meta`（JSON）到数据集级别的 sidecar 文件。


## 目录结构与产物规范（落实到仓库）

- 每个数据集在 `data/<DATASET>/` 占一文件夹：
  - 预处理脚本：`preprocess_<dataset>.py`
  - 预处理输出：
    - `data.pkl`（统一数据文件，建议保存为 `List[Tuple[dgl_graph, graph_label]]` 或 `List[Dict]`）
    - `train_index.json`, `val_index.json`, `test_index.json`（三分划分索引）
  - 其余可选：`README.md`（记录产物说明、统计与溯源）

- 加载器位于 `src/data/<dataset>_loader.py`，并在 `src/data/unified_data_factory.py` 注册（延迟导入工厂 `_lazy_import_loader` / `_register_all_loaders`）

- 加载器需要遵循 `BaseDataLoader` 抽象接口：
  - 从预处理目录读取四个文件；按三分索引切分
  - 构建/缓存整图张量：
    - 节点：`node_token_ids: LongTensor [N, 1]`、`node_type_id: LongTensor [N]`
    - 边：`edge_token_ids: LongTensor [E, 1]`、`edge_type_id: LongTensor [E]`
  - 实现批量/整图 API：`get_graph_node_token_ids`、`get_graph_edge_token_ids`、`get_node_tokens_bulk`、`get_edge_tokens_bulk` 等


## 数据集逐项说明与核验要点

以下“统计”来自我们对全量数据集的程序化核验（`foreign_dataset_files_to_convert/check_dgl_graphpred.py` 全量运行），仅列出与 token 化决策相关的关键点。

### 1) PROTEINS（TU）
- 加载：`TUDataset(name='PROTEINS')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique=3）；`node_attr` 为连续 1 维（cont1d，unique≈61）
  - 边：无 `edge_attr/edge_labels`
- 预处理决策：
  - 节点用 label：`token = node_labels`。
  - 边无属性：`token = 0`（常数）。

### 2) COLORS-3（TU，合成）
- 加载：`TUDataset(name='COLORS-3')`
- 实测统计（全量）：
  - 节点：`node_attr` 为 5 维多热（multihot），unique≈11（不同多热模式数）
  - 节点不提供 `node_labels`
  - 边：无 `edge_attr/edge_labels`
- 预处理决策：
  - 节点用 attr：将多/one-hot 向量映射为单一离散 ID，作为 `token`。
    - 建议实现：将二值向量视作 bit pattern 映射为整数（或使用稳定哈希/字典映射）。
  - 边无属性：`token = 0`。

实现状态：已完成
- 预处理脚本：`data/COLORS-3/preprocess_colors3.py`
  - 将每个节点的 `node_attr`（5 维二值 multi-hot）按 bit 编码映射为整数 → 写入 `g.ndata['node_token_ids']` 与 `g.ndata['node_type_id']`
  - 边字段缺失 → `g.edata['edge_token_ids']=0`、`edge_type_id=0`
  - 产物：`data/COLORS-3/{data.pkl, train_index.json, val_index.json, test_index.json}`
- 加载器：`src/data/colors3_loader.py`（已注册为 `colors3`）
  - 读取四个文件，提供统一接口与整图张量 API
  - 任务类型：图分类（`get_dataset_task_type() -> 'classification'`，`get_num_classes() -> 11`）

### 3) SYNTHETIC（TU，合成）
- 加载：`TUDataset(name='SYNTHETIC')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique=8）；`node_attr` 为连续 1 维（cont1d，unique≈29887）
  - 边：无 `edge_attr/edge_labels`
- 预处理决策：
  - 节点用 label：`token = node_labels`。
  - 边无属性：`token = 0`。

### 4) Mutagenicity（TU，分子）
- 加载：`TUDataset(name='Mutagenicity')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique=14）
  - 边：`edge_labels`（int，unique=3）
- 预处理决策：
  - 节点用 label：`token = node_labels`。
  - 边用 label：`token = edge_labels`。

### 5) COIL-DEL（TU，视觉图）
- 加载：`TUDataset(name='COIL-DEL')`
- 实测统计（全量）：
  - 节点：`node_attr` 为 2 维连续（float）；整列唯一和 `sum_unique_cols≈254`；按成对组合去重 `pair_unique≈14130`；若将两列整数化后相乘再去重，`unique≈4413`（单独脚本验证）
  - 边：`edge_labels`（int，unique=2）
- 预处理决策：
  - 节点用 attr：先将两列取整（int），以“乘积”作为节点 `token`（即 `(col0_int * col1_int)`）。
    - 说明：该策略在全量数据下得到 `unique≈4413`。
  - 边用 label：`token = edge_labels`。

实现提示：
- 预处理脚本需对 `node_attr` 两列做 `astype(np.int64)` 后取积；写入 `node_token_ids`/`node_type_id`
- 边侧从 `edge_labels` 直接写入 `edge_token_ids`/`edge_type_id`

### 6) DBLP_v1（TU，多图版）
- 加载：`TUDataset(name='DBLP_v1')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique≈41325）
  - 边：`edge_labels`（int，unique=3）
- 预处理决策：
  - 节点用 label：`token = node_labels`（注意唯一值数量较大）。
  - 边用 label：`token = edge_labels`。

实现提示：
- 节点侧直接 `g.ndata['node_token_ids']=node_labels.view(-1,1)`，`node_type_id=node_labels`
- 边侧 `g.edata['edge_token_ids']=edge_labels.view(-1,1)`，`edge_type_id=edge_labels`

### 7) DD（TU，蛋白质，规模较大）
- 加载：`TUDataset(name='DD')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique≈82）
  - 边：无 `edge_attr/edge_labels`
- 预处理决策：
  - 节点用 label：`token = node_labels`。
  - 边无属性：`token = 0`。

实现提示：同 DBLP_v1 节点侧；边侧填 0

### 8) TWITTER-Real-Graph-Partial（TU）
- 加载：`TUDataset(name='TWITTER-Real-Graph-Partial')`
- 实测统计（全量）：
  - 节点：`node_labels`（int，unique≈1323）
  - 边：无 `edge_attr/edge_labels`
- 预处理决策：
  - 节点用 label：`token = node_labels`。
  - 边无属性：`token = 0`。

实现提示：同 DBLP_v1 节点侧；边侧填 0


## COLORS-3 实现与使用（操作手册）

1) 预处理（会自动下载 TU 数据到 `~/.dgl/`）
```bash
python data/COLORS-3/preprocess_colors3.py
```

2) 加载处理后数据
```python
from config import ProjectConfig
from src.data.unified_data_factory import get_dataloader

cfg = ProjectConfig()
loader = get_dataloader('colors3', cfg)
train_data, val_data, test_data, y_tr, y_va, y_te = loader.load_data()

g = train_data[0]['dgl_graph']
node_tok = loader.get_graph_node_token_ids(g)  # [N,1] LongTensor
edge_tok = loader.get_graph_edge_token_ids(g)  # [E,1] LongTensor
```

3) 目录产物（生成后）
```
data/COLORS-3/
├── data.pkl
├── train_index.json
├── val_index.json
└── test_index.json
```


## 其它数据集的实现模板（步骤）

以 `PROTEINS` 为例，可复制 `data/COLORS-3/preprocess_colors3.py` 为模板：
- 放置到 `data/PROTEINS/preprocess_proteins.py`
- 节点：从 `node_labels`（int）写入 `node_token_ids`/`node_type_id`
- 边：缺失则 0；若存在 `edge_labels` 则写入对应 token 张量
- 切分：按固定 8/1/1 顺序切分，生成三份 JSON
- 保存：`data.pkl` + 三份 JSON

加载器模板：
- 新建 `src/data/proteins_loader.py`，复制 `src/data/colors3_loader.py` 基架并按需简化
- 在 `src/data/unified_data_factory.py` 注册：
  - `_lazy_import_loader()` 添加 `elif loader_name == "proteins": from .proteins_loader import PROTEINSLoader; return PROTEINSLoader`
  - `_register_all_loaders()` 列表增加 `"proteins"`


## 测试与验证清单

- 预处理阶段：
  - 运行预处理脚本，确认生成 `data.pkl` 与三份划分 JSON
  - 随机抽样加载 `data.pkl`，检视首个样本的 `dgl_graph` 是否包含 `node_token_ids`/`edge_token_ids`

- 加载阶段：
  - 通过 `get_dataloader('<name>', cfg)` 加载，取得三分数据
  - 取一个 `graph = sample['dgl_graph']`，验证：
    - `loader.get_graph_node_token_ids(graph)` 形状 `[N,1]`，整型
    - `loader.get_graph_edge_token_ids(graph)` 形状 `[E,1]`，整型
  - 对于无边属性的数据集，边 token 应全 0

- 统计对齐（可选）：
  - 对比 `foreign_dataset_files_to_convert/check_dgl_graphpred.py` 的 unique 统计（全量）与当前实现的离散空间是否一致/可解释


## 贡献约定

- 严格遵循本文档的 token 化决策与落地路径（字段名、产物文件名不可改变）
- 代码风格与接口：保持与 `BaseDataLoader` 约定一致，不做隐式回退
- PR 内容：
  - `data/<NAME>/preprocess_<name>.py` 与产物格式说明
  - `src/data/<name>_loader.py` 与 `unified_data_factory.py` 的注册
  - 简短的 README 或在本文档中追加“实现状态”小节与使用说明


## 统一实现要点

- 目标字段
  - 节点侧：写入 `g.ndata['token']`（`torch.int64`），长度等于节点数。
  - 边侧：写入 `g.edata['token']`（`torch.int64`），长度等于边数。

- 标签到 token（通用于节点/边）
  - 若存在 `*_labels` 且为整型 1D：直接用作 `token`。

- 多/one-hot 到 token（节点侧示例，边侧同理）
  - 处理 `node_attr` 为二值矩阵时：
    - one-hot：`token = argmax(node_attr, axis=1)`。
    - multihot：将整行二值向量转换为整数 ID（例如将 bit 向量视作二进制数；或将 tuple 作为键做字典映射到新 ID）。
  - 要求映射稳定、可逆（至少可追踪）：建议在数据集级保存 `pattern -> id` 字典。

- 连续到 token（COIL-DEL 特例）
  - 对 `node_attr` 为两列连续值：先转为整数（如 `astype(np.int64)`，保持与上游统计一致），取乘积作为 `token`。
  - 若未来需要其它离散化方案（如分箱/哈希），可在此位置替换。

- 边侧常数填充
  - 若不存在 `edge_attr` 与 `edge_labels`：
    - 创建 `g.edata['token'] = torch.zeros(num_edges, dtype=torch.int64)`。

- 保留原字段
  - 不覆盖原始 `node_labels`/`node_attr`/`edge_labels`/`edge_attr`，以便复查与追踪。

- 批处理与缓存
  - 可为每个数据集编写独立的 `preprocess_<dataset>.py`，内部按本方案生成 `token` 后序列化保存（例如使用 DGL 的 `save_graphs`）。
  - 也可写一个统一入口，根据 `dataset_name` 分发到对应规则。


## 附：示例伪代码片段

```python
# 节点：label -> token
if 'node_labels' in g.ndata and g.ndata['node_labels'].ndim == 1:
    g.ndata['token'] = g.ndata['node_labels'].long()

# 节点：one-hot -> token
elif 'node_attr' in g.ndata and g.ndata['node_attr'].dim() == 2 and is_one_hot(g.ndata['node_attr']):
    g.ndata['token'] = g.ndata['node_attr'].argmax(dim=1).long()

# 节点：multihot -> token（示意：bit向量编码）
elif 'node_attr' in g.ndata and g.ndata['node_attr'].dim() == 2 and is_binary(g.ndata['node_attr']):
    g.ndata['token'] = binary_row_to_int(g.ndata['node_attr'])  # 自定义稳定映射

# 节点：COIL-DEL 特例（两列连续取积）
elif dataset_name == 'COIL-DEL' and 'node_attr' in g.ndata and g.ndata['node_attr'].shape[1] == 2:
    ai = g.ndata['node_attr'].to(torch.int64)
    g.ndata['token'] = (ai[:, 0] * ai[:, 1]).long()

# 边：优先用 edge_labels，否则常数 0
if 'edge_labels' in g.edata and g.edata['edge_labels'].ndim == 1:
    g.edata['token'] = g.edata['edge_labels'].long()
else:
    g.edata['token'] = torch.zeros(g.num_edges(), dtype=torch.int64)
```


## 备注

- 统计值来自我们工具全量运行，可能因上游库版本或数据镜像差异略有出入，但不影响规则选择与实现落地。
- 若后续需要支持更多 DGL 集成数据集，可按“先 label、后 attr（one/multihot -> 单 ID；连续 -> 明确策略）”的统一流程扩展。




## 附：程序化原始输出

```bash
(base) gzy@gamma-a800:~/py/tokenizerGraph$ /home/gzy/miniconda3/envs/pthgnn/bin/python3 -u /home/gzy/py/tokenizerGraph/foreign_dataset_files_to_convert/check_dgl_graphpred.py | cat
/home/gzy/miniconda3/envs/pthgnn/lib/python3.10/site-packages/outdated/__init__.py:36: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import parse_version
开始分析数据集：PROTEINS ...
================================================================================
数据集: PROTEINS
图数量: 1113
节点数统计: min=4, max=620, mean=39.06, std=45.76
节点数分位数: p50=26, p90=81, p95=111, p99=206
边数统计: min=10, max=2098, mean=145.63, std=169.20
边数分位数: p50=98, p90=300, p95=408, p99=777
节点特征: exists=True, keys=['_ID', 'node_attr', 'node_labels'], main_key=node_attr, dim=1, dtype=float64, 离散样式=False, onehot样式=False
节点特征示例(前3行x前8列):
  [23.0]
  [10.0]
  [25.0]
边特征: exists=False, keys=['_ID'], main_key=None, dim=None, dtype=None, 离散样式=None, onehot样式=None
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=3, notes=None
  节点 label 路径: key=node_labels, unique_tokens=3, mode=int, notes=None
  节点 attr  路径: key=node_attr, unique_tokens=61, mode=cont1d, notes=None
边 token 源(自动选择): source=None, key=None, unique_tokens=None, notes=None
  边   label 路径: key=None, unique_tokens=None, mode=None, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：COLORS-3 ...
================================================================================
数据集: COLORS-3
图数量: 10500
节点数统计: min=4, max=200, mean=61.31, std=60.52
节点数分位数: p50=24, p90=164, p95=182, p99=197
边数统计: min=8, max=794, mean=182.05, std=187.34
边数分位数: p50=78, p90=478, p95=570, p99=692
节点特征: exists=True, keys=['_ID', 'node_attr'], main_key=node_attr, dim=5, dtype=int64, 离散样式=True, onehot样式=False
节点特征示例(前3行x前8列):
  [1, 0, 1, 0, 0]
  [0, 0, 0, 1, 0]
  [0, 1, 0, 0, 0]
边特征: exists=False, keys=['_ID'], main_key=None, dim=None, dtype=None, 离散样式=None, onehot样式=None
图标签: shape=[1], dtype=float64
节点 token 源(自动选择): source=attr, key=node_attr, unique_tokens=11, notes=None
  节点 label 路径: key=None, unique_tokens=None, mode=None, notes=None
  节点 attr  路径: key=node_attr, unique_tokens=11, mode=multihot, notes=None
边 token 源(自动选择): source=None, key=None, unique_tokens=None, notes=None
  边   label 路径: key=None, unique_tokens=None, mode=None, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：SYNTHETIC ...
================================================================================
数据集: SYNTHETIC
图数量: 300
节点数统计: min=100, max=100, mean=100.00, std=0.00
节点数分位数: p50=100, p90=100, p95=100, p99=100
边数统计: min=392, max=392, mean=392.00, std=0.00
边数分位数: p50=392, p90=392, p95=392, p99=392
节点特征: exists=True, keys=['_ID', 'node_attr', 'node_labels'], main_key=node_attr, dim=1, dtype=float64, 离散样式=False, onehot样式=False
节点特征示例(前3行x前8列):
  [-0.025829]
  [-1.002684]
  [0.887984]
边特征: exists=False, keys=['_ID'], main_key=None, dim=None, dtype=None, 离散样式=None, onehot样式=None
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=8, notes=None
  节点 label 路径: key=node_labels, unique_tokens=8, mode=int, notes=None
  节点 attr  路径: key=node_attr, unique_tokens=29887, mode=cont1d, notes=None
边 token 源(自动选择): source=None, key=None, unique_tokens=None, notes=None
  边   label 路径: key=None, unique_tokens=None, mode=None, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：Mutagenicity ...
================================================================================
数据集: Mutagenicity
图数量: 4337
节点数统计: min=4, max=417, mean=30.32, std=20.12
节点数分位数: p50=27, p90=48, p95=62, p99=94
边数统计: min=6, max=224, mean=61.54, std=33.64
边数分位数: p50=56, p90=100, p95=126, p99=193
节点特征: exists=True, keys=['_ID', 'node_labels'], main_key=node_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=False
节点特征示例(前3行x前8列):
  [0]
  [0]
  [0]
边特征: exists=True, keys=['_ID', 'edge_labels'], main_key=edge_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=True
边特征示例(前3行x前8列):
  [0]
  [0]
  [1]
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=14, notes=None
  节点 label 路径: key=node_labels, unique_tokens=14, mode=int, notes=None
  节点 attr  路径: key=None, unique_tokens=None, mode=None, notes=None
边 token 源(自动选择): source=label, key=edge_labels, unique_tokens=3, notes=None
  边   label 路径: key=edge_labels, unique_tokens=3, mode=int, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：COIL-DEL ...
================================================================================
数据集: COIL-DEL
图数量: 3900
节点数统计: min=3, max=77, mean=21.54, std=13.22
节点数分位数: p50=20, p90=38, p95=47, p99=63
边数统计: min=6, max=444, mean=108.47, std=76.94
边数分位数: p50=96, p90=204, p95=258, p99=356
节点特征: exists=True, keys=['_ID', 'node_attr'], main_key=node_attr, dim=2, dtype=float64, 离散样式=False, onehot样式=False
节点特征示例(前3行x前8列):
  [7.0, 29.0]
  [34.0, 46.0]
  [45.0, 45.0]
边特征: exists=True, keys=['_ID', 'edge_labels'], main_key=edge_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=True
边特征示例(前3行x前8列):
  [0]
  [1]
  [1]
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=attr, key=node_attr, unique_tokens=254, notes=sum(unique(col0), unique(col1))
  节点 label 路径: key=None, unique_tokens=None, mode=None, notes=None
  节点 attr  路径: key=node_attr, unique_tokens=254, mode=coil_del_2col_float, notes=sum_unique_cols=254; pair_unique=14130
边 token 源(自动选择): source=label, key=edge_labels, unique_tokens=2, notes=None
  边   label 路径: key=edge_labels, unique_tokens=2, mode=int, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：DBLP_v1 ...
================================================================================
数据集: DBLP_v1
图数量: 19456
节点数统计: min=2, max=39, mean=10.48, std=8.54
节点数分位数: p50=7, p90=24, p95=30, p99=37
边数统计: min=2, max=336, mean=39.29, std=39.26
边数分位数: p50=28, p90=94, p95=124, p99=178
节点特征: exists=True, keys=['_ID', 'node_labels'], main_key=node_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=False
节点特征示例(前3行x前8列):
  [0]
  [1]
  [2]
边特征: exists=True, keys=['_ID', 'edge_labels'], main_key=edge_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=True
边特征示例(前3行x前8列):
  [0]
  [0]
  [0]
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=41325, notes=None
  节点 label 路径: key=node_labels, unique_tokens=41325, mode=int, notes=None
  节点 attr  路径: key=None, unique_tokens=None, mode=None, notes=None
边 token 源(自动选择): source=label, key=edge_labels, unique_tokens=3, notes=None
  边   label 路径: key=edge_labels, unique_tokens=3, mode=int, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：DD ...
================================================================================
数据集: DD
图数量: 1178
节点数统计: min=30, max=5748, mean=284.32, std=272.00
节点数分位数: p50=241, p90=501, p95=589, p99=892
边数统计: min=126, max=28534, mean=1431.32, std=1387.81
边数分位数: p50=1221, p90=2577, p95=3095, p99=4954
节点特征: exists=True, keys=['_ID', 'node_labels'], main_key=node_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=False
节点特征示例(前3行x前8列):
  [3]
  [1]
  [9]
边特征: exists=False, keys=['_ID'], main_key=None, dim=None, dtype=None, 离散样式=None, onehot样式=None
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=82, notes=None
  节点 label 路径: key=node_labels, unique_tokens=82, mode=int, notes=None
  节点 attr  路径: key=None, unique_tokens=None, mode=None, notes=None
边 token 源(自动选择): source=None, key=None, unique_tokens=None, notes=None
  边   label 路径: key=None, unique_tokens=None, mode=None, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
开始分析数据集：TWITTER-Real-Graph-Partial ...
================================================================================
数据集: TWITTER-Real-Graph-Partial
图数量: 144033
节点数统计: min=2, max=14, mean=4.03, std=1.69
节点数分位数: p50=4, p90=6, p95=7, p99=9
边数统计: min=2, max=116, mean=9.96, std=9.15
边数分位数: p50=6, p90=22, p95=30, p99=42
节点特征: exists=True, keys=['_ID', 'node_labels'], main_key=node_labels, dim=1, dtype=int64, 离散样式=True, onehot样式=False
节点特征示例(前3行x前8列):
  [0]
  [1]
  [2]
边特征: exists=False, keys=['_ID', 'node_labels'], main_key=None, dim=None, dtype=None, 离散样式=None, onehot样式=None
图标签: shape=[1], dtype=int64
节点 token 源(自动选择): source=label, key=node_labels, unique_tokens=1323, notes=None
  节点 label 路径: key=node_labels, unique_tokens=1323, mode=int, notes=None
  节点 attr  路径: key=None, unique_tokens=None, mode=None, notes=None
边 token 源(自动选择): source=None, key=None, unique_tokens=None, notes=None
  边   label 路径: key=None, unique_tokens=None, mode=None, notes=None
  边   attr  路径: key=None, unique_tokens=None, mode=None, notes=None
================================================================================
全部数据集分析完成。

```
