## 目标与筛选标准

- **任务级别**: 图级（graph-level）分类/回归。
- **特征硬指标**: 原始特征不可为连续实值。允许：
  - 无特征；
  - 单维离散特征；
  - 多维 one-hot（语义为离散）；
  - 多离散字段（需标注）。
- **规模期望**: 单图边数在“几十–几百”范围优先。
- **优先方向**: 非分子图数据集；分子图只额外补充 1 个权威数据集。

参考：
- DGL 图级任务数据集总览：[Graph Prediction Datasets](https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.data.html#graph-prediction-datasets)
- TU 数据集官网统计（包含平均边数与是否有离散/连续属性）：[TU Datasets](https://chrsmrrs.github.io/datasets/docs/datasets/)
- DGL `TUDataset` 说明：[dgl.data.TUDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)


## 推荐（非分子，优先验证）

### IMDB-BINARY（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: 电影演员的 ego 网络，节点为演员，边为同演出连边，图标签对应电影类型（两类）。
- **规模**: Graphs=1000, Classes=2, Avg. Nodes≈19.77, Avg. Edges≈96.53
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−（原始无特征，无连续）
- **权威性**: 常见的社交/合作网络图分类基线，广泛用于 GNN 论文。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='IMDB-BINARY')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/IMDB-BINARY.zip) · [DGL 文档](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)

### IMDB-MULTI（TU）
- **任务**: 图分类（3 类）
- **用途/描述**: 与 IMDB-BINARY 类似，为三类电影类型分类。
- **规模**: Graphs=1500, Classes=3, Avg. Nodes≈13.00, Avg. Edges≈65.94
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 与 IMDB-BINARY 并列的常用基线。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='IMDB-MULTI')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/IMDB-MULTI.zip)

### twitch_egos（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: Twitch 平台的 ego 网络，节点为用户/频道，边为交互关系，二分类任务。
- **规模**: Graphs=127094, Classes=2, Avg. Nodes≈29.67, Avg. Edges≈86.59
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 大规模社交平台 ego 图集合，应用广泛。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='twitch_egos')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/twitch_egos.zip)

### reddit_threads（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: Reddit 讨论串的线程图，节点为发帖/回复用户，边为回复/交互。
- **规模**: Graphs=203088, Classes=2, Avg. Nodes≈23.93, Avg. Edges≈24.99（偏小但仍属“几十”下沿）
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 常见的社交讨论线程图集合。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='reddit_threads')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/reddit_threads.zip)

### TRIANGLES（TU，合成）
- **任务**: 图分类（10 类）
- **用途/描述**: 合成图，按图中三角形模式数量/结构进行分类，测试 GNN 对局部结构的识别能力。
- **规模**: Graphs=45000, Classes=10, Avg. Nodes≈20.85, Avg. Edges≈32.74
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 合成基准，常用于表达力与模式识别评测。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='TRIANGLES')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/TRIANGLES.zip)

### COLORS-3（TU，合成）
- **任务**: 图分类（11 类）
- **用途/描述**: 合成图的着色/模式识别任务。
- **规模**: Graphs=10500, Classes=11, Avg. Nodes≈61.31, Avg. Edges≈91.03
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=+ (4 维 one-hot), Edge Attr.=−（离散 one-hot，满足硬指标）
- **权威性**: 合成基准，常用于评估模型对着色/全局模式的建模能力。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='COLORS-3')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/COLORS-3.zip)

### MiniGCDataset（DGL，合成）
- **任务**: 图分类（可配置图生成规则）
- **用途/描述**: 用于测试 GNN 对经典图结构/生成过程的判别能力。
- **规模**: n_graphs 可设定；常设“几十–几百”边
- **特征/标签**: 默认无节点/边特征（不含连续属性），无节点/边标签字段
- **权威性**: DGL 官方提供的合成基准，便于可控实验。
- **加载**:
  ```python
  from dgl.data import MiniGCDataset
  ds = MiniGCDataset(n_graphs=1000, min_num_v=10, max_num_v=20)
  ```
- **链接**: [DGL MiniGCDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.MiniGCDataset.html#dgl.data.MiniGCDataset)

### BA2MotifDataset（DGL，合成）
- **任务**: 图分类/可解释性评测（motif 检测）
- **用途/描述**: 由 BA 图注入 2 种 motif，用于评估解释器/模型对 motif 的敏感性。
- **规模**: n_graphs 可设定；常见“几十–百级”边
- **特征/标签**: Node Labels=+（角色/类型，离散），Edge Labels=−，Node/Edge Attr.=−
- **权威性**: 源自解释性论文基准，使用广泛。
- **加载**:
  ```python
  from dgl.data import BA2MotifDataset
  ds = BA2MotifDataset(1000)
  ```
- **链接**: [DGL BA2MotifDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.BA2MotifDataset.html#dgl.data.BA2MotifDataset)


## 可选（更大一些的“几百”边）

### REDDIT-BINARY（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: Reddit 讨论帖的线程图，二分类标签（不同板块/主题）。
- **规模**: Graphs=2000, Classes=2, Avg. Nodes≈429.63, Avg. Edges≈497.75
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 与 IMDB 并列的经典社交图分类基线，使用频繁。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='REDDIT-BINARY')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip)

### REDDIT-MULTI-12K（TU）
- **任务**: 图分类（11 类）
- **用途/描述**: Reddit 讨论帖多分类任务（多个子版面/主题）。
- **规模**: Graphs=11929, Classes=11, Avg. Nodes≈391.41, Avg. Edges≈456.89
- **特征/标签**: Node Labels=−, Edge Labels=−, Node Attr.=−, Edge Attr.=−
- **权威性**: 大规模 Reddit 基线，用于评估模型扩展性与泛化。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='REDDIT-MULTI-12K')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip)


## 分子类（额外补充 1 个权威）

> 你已实现 `QM9Dataset`、`QM9EdgeDataset`、`ZINCDataset`，此处仅补充 1 个常用权威分子图数据集以便横向对照。

### Mutagenicity（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: 分子是否具有致突变性（mutagenic）的二分类任务。
- **规模**: Graphs=4337, Classes=2, Avg. Nodes≈30.32, Avg. Edges≈30.77
- **特征/标签**: Node Labels=+（原子类型等离散）, Edge Labels=+（键类型等离散）, Node/Edge Attr.=−（无连续）
- **权威性**: 传统分子图分类基准之一，常与 MUTAG、NCI 系列并列出现。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='Mutagenicity')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip)


## 快速下载与格式核验建议

- 统一以 DGL 接口加载，逐一检查：
  - 图规模（边数分布是否在“几十–几百”）。
  - `ndata` / `edata` 各字段 dtype 是否为整型或空；若为 one-hot 浮点但语义离散，可接受并记录映射方案。
- 示例脚本：
  ```python
  import numpy as np
  from dgl.data import TUDataset, MiniGCDataset, BA2MotifDataset

  def inspect_dataset(ds, max_samples=1000):
      m = min(len(ds), max_samples)
      edge_counts = []
      for i in range(m):
          g, y = ds[i]
          edge_counts.append(g.num_edges())
      print('graphs:', m, 'edges(min/med/max)=', int(np.min(edge_counts)), int(np.median(edge_counts)), int(np.max(edge_counts)))

  # 例子
  inspect_dataset(TUDataset(name='IMDB-BINARY'))
  inspect_dataset(TUDataset(name='IMDB-MULTI'))
  inspect_dataset(TUDataset(name='TRIANGLES'))
  inspect_dataset(TUDataset(name='COLORS-3'))
  inspect_dataset(TUDataset(name='twitch_egos'))
  inspect_dataset(TUDataset(name='REDDIT-BINARY'))
  inspect_dataset(TUDataset(name='REDDIT-MULTI-12K'))
  inspect_dataset(TUDataset(name='reddit_threads'))
  inspect_dataset(MiniGCDataset(200, 10, 20))
  inspect_dataset(BA2MotifDataset(200))
  inspect_dataset(TUDataset(name='Mutagenicity'))
  ```


## 备注与排除

- 明确排除（原始含连续实值）
  - `MNISTSuperPixelDataset`、`CIFAR10SuperPixelDataset`（像素/坐标为连续）
  - `FakeNewsDataset`（文本/画像衍生连续向量常见）
  - 部分 TU 子集（如 ENZYMES/PROTEINS/DD 等）包含连续节点属性时不符合硬指标
- 规模偏大的可谨慎选用或二次筛样：
  - `COLLAB`（常显著大于“几百”边）


## 索引与链接
- DGL 图级任务列表：[`dgl.data`](https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.data.html#graph-prediction-datasets)
- DGL `TUDataset`：[`文档`](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)
- TU 数据集官网：[`统计总表`](https://chrsmrrs.github.io/datasets/docs/datasets/)
