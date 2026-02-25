## 节点可离散成 Token 的图级任务数据集清单（DGL/TU）

本清单仅收录：节点可由离散字段（节点标签或离散/one-hot 节点属性）映射为单一离散值（token）。边特征/标签可无；但若存在且为离散，会注明。图级任务（classification/regression）。

- 术语澄清：
  - **图标签（graph label）**：每个图一条监督信号（分类/回归）。
  - **节点标签（node label）/节点属性（node attr.）**：随图提供的节点级字段，可作为节点特征。若为离散/one-hot，则可以直接 token 化。
  - **边标签（edge label）/边属性（edge attr.）**：同理于边级。
- 硬指标：节点侧需存在离散信息（标签或 one-hot 属性），可合并映射到单个离散 ID。
- 规模偏好：单图边数在“几十–几百”优先。

参考：
- DGL 图级任务数据集总览：[Graph Prediction Datasets](https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.data.html#graph-prediction-datasets)
- TU 数据集官网统计（含 Graphs/Classes/Avg.Nodes/Avg.Edges/Labels/Attr.）：[TU Datasets](https://chrsmrrs.github.io/datasets/docs/datasets/)
- DGL `TUDataset` 说明：[dgl.data.TUDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)


## 非分子（优先）

### PROTEINS（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: 蛋白质二级结构形成的图，节点表示二次结构成分/残基（文献常视为离散类型），图标签为蛋白是否属于某类。
- **规模（TU 统计）**: Graphs≈1113, Classes=2, Avg. Nodes≈39, Avg. Edges≈73（数量级：几十–一百）
- **节点/边字段**: Node Labels=+（离散），Edge Labels=−，Node Attr.=−，Edge Attr.=−
- **权威性**: 经典 TU 基准，GIN 等论文常用（与 NCI1/MUTAG 并列）。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='PROTEINS')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [DGL TUDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)
- **备注**: 节点标签可直接映射为 token；规模契合“几十–几百边”。

### COLORS-3（TU，合成）
- **任务**: 图分类（11 类）
- **用途/描述**: 合成图着色/模式识别任务，节点属性为 4 维 one-hot（离散语义）。
- **规模（TU 统计）**: Graphs=10500, Classes=11, Avg. Nodes≈61.31, Avg. Edges≈91.03
- **节点/边字段**: Node Labels=−，Edge Labels=−，Node Attr.=+ (4, one-hot)，Edge Attr.=−
- **权威性**: 合成基准，广泛用于检验模型对着色/模式的能力。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='COLORS-3')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/COLORS-3.zip)
- **备注**: one-hot 可直接作为 token（或映射到单一离散 ID）。

### SYNTHETIC（TU，合成）
- **任务**: 图分类（二分类）
- **用途/描述**: 合成数据集，评估对结构/生成过程的判别能力。
- **规模（TU 统计）**: Graphs=300, Classes=2, Avg. Nodes=100.00, Avg. Edges=196.00
- **节点/边字段**: Node Labels=+（离散），Edge Labels=−，Node Attr.=+ (1)，Edge Attr.=−
- **权威性**: 经典 TU 合成基准之一。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='SYNTHETIC')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/SYNTHETIC.zip)
- **备注**: Node Labels 已满足 token 要求；Node Attr.(1) 一般可忽略或与标签合并为单离散值。

### BA2MotifDataset（DGL，合成）
- **任务**: 图分类/可解释性评测（motif 检测）
- **用途/描述**: BA 图注入 2 种 motif，节点带离散“角色”标签，用于解释器与模型对 motif 的敏感性评估。
- **规模**: n_graphs 可设定；常见“几十–百级”边
- **节点/边字段**: Node Labels=+（离散角色），Edge Labels=−，Node/Edge Attr.=−
- **权威性**: 源自解释性研究的标准基准，使用广泛。
- **加载**:
  ```python
  from dgl.data import BA2MotifDataset
  ds = BA2MotifDataset(1000)
  ```
- **链接**: [DGL BA2MotifDataset](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.BA2MotifDataset.html#dgl.data.BA2MotifDataset)


## 分子（仅补充 1 个权威）

> 你已实现 QM9/QM9Edge/ZINC；此处仅补充 1 个权威分子图，满足节点/边离散标签、便于 token 化。

### Mutagenicity（TU）
- **任务**: 图分类（二分类）
- **用途/描述**: 判定分子是否致突变（mutagenic）。
- **规模（TU 统计）**: Graphs=4337, Classes=2, Avg. Nodes≈30.32, Avg. Edges≈30.77
- **节点/边字段**: Node Labels=+（原子类型等离散），Edge Labels=+（键类型等离散），Node/Edge Attr.=−
- **权威性**: 传统分子图基准之一，极常用。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='Mutagenicity')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · [ZIP](https://www.chrsmrrs.com/graphkerneldatasets/Mutagenicity.zip)


## 说明：为何 IMDB/REDDIT 等未入选
- IMDB-BINARY/IMDB-MULTI、REDDIT-BINARY/REDDIT-MULTI-12K、twitch_egos、reddit_threads 等：原始不提供节点标签/属性（TU 表为 "-"），无法直接对节点进行离散 token 化（尽管图标签存在且可做图分类）。这些数据集在 GNN 研究中常通过“常数/度数/位置编码”等替代特征进行训练，但与本清单的“节点必须可离散 token 化”的硬指标不符。


## 节点离散映射建议
- 对含节点标签的数据集（如 PROTEINS、SYNTHETIC、BA2Motif、Mutagenicity）：
  - 直接使用标签 ID 作为 token。
- 对 one-hot 节点属性的数据集（如 COLORS-3）：
  - 若唯一 one-hot 维度：用 argmax 得到单一离散 ID；
  - 若多离散字段：使用哈希或多字段组合映射到单一 token 空间（保持可逆/可追踪）。


## 统一核验脚本（抽样）
```python
import numpy as np
from dgl.data import TUDataset

candidates = [
    ('PROTEINS', True),
    ('COLORS-3', True),
    ('SYNTHETIC', True),
    ('Mutagenicity', True),
]

def inspect_tu(name, max_samples=1000):
    ds = TUDataset(name=name)
    m = min(len(ds), max_samples)
    edges = []
    for i in range(m):
        g, y = ds[i]
        edges.append(g.num_edges())
    print(name, 'graphs:', len(ds), 'edges(min/med/max)=', int(np.min(edges)), int(np.median(edges)), int(np.max(edges)))

for name, _ in candidates:
    inspect_tu(name)
```


## 索引与链接
- DGL 图级任务列表：[`dgl.data`](https://www.dgl.ai/dgl_docs/en/2.2.x/api/python/dgl.data.html#graph-prediction-datasets)
- DGL `TUDataset`：[`文档`](https://www.dgl.ai/dgl_docs/en/2.2.x/generated/dgl.data.TUDataset.html#dgl.data.TUDataset)
- TU 数据集官网：[`统计总表`](https://chrsmrrs.github.io/datasets/docs/datasets/)


## 新增：可纳入的数据集（补充）

### COIL-DEL（TU，视觉图，离散可 token 化）
- **任务**: 图分类（对象类别）
- **用途/描述**: 源自 COIL 图像库的图构建版本；节点为图像分块/超像素等离散单元，图标签为对象类别。
- **规模**: 参考 TU 统计（常见规模在“几十–百级”节点与边；具体以下载版本为准）。
- **节点/边字段**: Node Labels=+（离散），Edge Labels=−，Node Attr.=（如存在，要求为离散/one-hot；若提供连续几何字段，可忽略以满足硬指标），Edge Attr.=−
- **权威性**: TU 视觉类基准之一，常与 COIL-RAG 并列出现。
- **纳入结论**: 符合“节点可离散 token 化”要求；如出现连续几何字段请在预处理时关闭/忽略。
- **加载**:
  ```python
  from dgl.data import TUDataset
  ds = TUDataset(name='COIL-DEL')
  ```
- **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · DGL `TUDataset`

- **DBLP_v1（TU，多图版本）**
  - **说明**: 这里指 TU 提供的多图版 `DBLP_v1`（而非单大图的原始 DBLP 网络）。该版本以多图切分组织，可直接纳入图级基准。
  - **节点/边字段**: 节点含离散字段（如类别/主题/类型等），满足“节点可离散 token 化”。
  - **加载**:
    ```python
    from dgl.data import TUDataset
    ds = TUDataset(name='DBLP_v1')
    ```
  - **链接**: [TU 统计](https://chrsmrrs.github.io/datasets/docs/datasets/) · DGL `TUDataset`

- **DD（已纳入）**
  - 已在“新增：可纳入的数据集”中列出，标注为“可选（规模较大）”。

- **Tox21（纳入）**
  - **决定**: 纳入（节点/边均为离散，满足 token 化；为分子权威数据集之一，适合作为非 QM9/ZINC 的补充）。
  - **获取**: 通过 MoleculeNet/DeepChem 或 Tox21 官方下载后自行转图。
