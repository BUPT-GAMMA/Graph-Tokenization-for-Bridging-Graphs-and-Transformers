这份文档的定位是**仓库内部数据集总览**，我采用了与发布文档一致的清晰层级结构进行了重新排版。所有技术细节（张量维度、侧文件、重建入口等）已 100% 精确保留，同时去除了原有的冗余表述，使其更适合作为开发者查阅的 Reference。

直接复制以下 Markdown 即可：

***

# 数据集总览文档

此文档记录当前仓库内各数据集的主题、评估指标（Metric）、来源、实际存储格式及读取逻辑。

---

## 1. `aqsol`
- **基本信息**
  - **主题:** 分子溶解度回归
  - **Metric:** MAE
  - **来源:** AqSol raw zip + dictionary pipeline
  - **参考处理脚本:** `data/aqsol/prepare_aqsol_raw.py`
  - **样本规模:** 9,823
- **文件与入口**
  - **主数据文件:** `data/aqsol/data.pkl`
  - **相关侧文件:** `smiles_1_direct.txt`, `smiles_2_explicit_h.txt`, `smiles_3_addhs.txt`, `smiles_4_addhs_explicit_h.txt`, `conversion_stats.json`
  - **统一 Loader 入口:** `get_dataloader("aqsol", cfg).load_data()`
  - **重建入口:** `src.data.loader.aqsol_loader.AQSOLoader._load_data_file`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, float]]` (单样本: `tuple[DGLGraph, float]`)
  - **Label 类型:** `float` (`value_type=float`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接包一层 sample dict)
  - **节点特征 (ndata):** `['feat']`，细节: `{'feat': 'shape=(70,), dtype=torch.int64'}`
  - **边特征 (edata):** `['feat']`，细节: `{'feat': 'shape=(136,), dtype=torch.int64'}`
- **直接读取示例:**
  ```python
  import pickle
  with open('data/aqsol/data.pkl', 'rb') as f:
      data = pickle.load(f)
  ```

## 2. `coildel`
- **基本信息**
  - **主题:** 视觉目标图分类
  - **Metric:** Accuracy
  - **来源:** TU `COIL-DEL`
  - **参考处理脚本:** `data/coildel/preprocess_coil_del.py`
  - **样本规模:** 3,900
- **文件与入口**
  - **主数据文件:** `data/coildel/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("coildel", cfg).load_data()`
  - **重建入口:** `src.data.loader.coildel_loader.COILDELLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_attr', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(26,), int64 | `node_attr`: shape=(26, 2), float64 | `node_token_ids`: shape=(26, 1), int64 | `node_type_id`: shape=(26,), int64
  - **边特征 (edata):** `['_ID', 'edge_labels', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(138,), int64 | `edge_labels`: shape=(138, 1), int64 | `edge_token_ids`: shape=(138, 1), int64 | `edge_type_id`: shape=(138,), int64
- **直接读取示例:** 同上，替换路径为 `data/coildel/data.pkl` 即可。

## 3. `colors3`
- **基本信息**
  - **主题:** 彩色图结构分类
  - **Metric:** Accuracy
  - **来源:** TU `COLORS-3`
  - **参考处理脚本:** `data/colors3/preprocess_colors3.py`
  - **样本规模:** 10,500
- **文件与入口**
  - **主数据文件:** `data/colors3/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("colors3", cfg).load_data()`
  - **重建入口:** `src.data.loader.colors3_loader.COLORS3Loader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_attr', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(6,), int64 | `node_attr`: shape=(6, 5), int64 | `node_token_ids`: shape=(6, 1), int64 | `node_type_id`: shape=(6,), int64
  - **边特征 (edata):** `['_ID', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(12,), int64 | `edge_token_ids`: shape=(12, 1), int64 | `edge_type_id`: shape=(12,), int64

## 4. `dblp`
- **基本信息**
  - **主题:** 学术网络图分类
  - **Metric:** Accuracy
  - **来源:** TU `DBLP_v1`
  - **参考处理脚本:** `data/dblp/preprocess_dblp_v1.py`
  - **样本规模:** 19,456
- **文件与入口**
  - **主数据文件:** `data/dblp/data.pkl`
  - **相关侧文件:** `summary.json`
  - **统一 Loader 入口:** `get_dataloader("dblp", cfg).load_data()`
  - **重建入口:** `src.data.loader.dblp_loader.DBLPLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(11,), int64 | `node_labels`: shape=(11, 1), int64 | `node_token_ids`: shape=(11, 1), int64 | `node_type_id`: shape=(11,), int64
  - **边特征 (edata):** `['_ID', 'edge_labels', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(20,), int64 | `edge_labels`: shape=(20, 1), int64 | `edge_token_ids`: shape=(20, 1), int64 | `edge_type_id`: shape=(20,), int64

## 5. `dd`
- **基本信息**
  - **主题:** 蛋白质图分类
  - **Metric:** Accuracy
  - **来源:** TU `DD`
  - **参考处理脚本:** `data/dd/preprocess_dd.py`
  - **样本规模:** 1,178
- **文件与入口**
  - **主数据文件:** `data/dd/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("dd", cfg).load_data()`
  - **重建入口:** `src.data.loader.dd_loader.DDLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(327,), int64 | `node_labels`: shape=(327, 1), int64 | `node_token_ids`: shape=(327, 1), int64 | `node_type_id`: shape=(327,), int64
  - **边特征 (edata):** `['_ID', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(1798,), int64 | `edge_token_ids`: shape=(1798, 1), int64 | `edge_type_id`: shape=(1798,), int64

## 6. `mnist_raw`
- **基本信息**
  - **主题:** MNIST 像素网格图分类
  - **Metric:** Accuracy
  - **来源:** `torchvision.datasets.MNIST`
  - **参考处理脚本:** `data/mnist_raw/prepare.py`
  - **样本规模:** 70,000
- **文件与入口**
  - **主数据文件:** `data/mnist_raw/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("mnist_raw", cfg).load_data()`
  - **重建入口:** `src.data.loader.mnist_raw_loader.MNISTRawDataLoader._load_raw_pairs + _image_to_dgl`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[numpy.ndarray, int]]` (单样本: `tuple[numpy.ndarray, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `numpy.ndarray`
  - **存储说明:** Pkl 中存储的是原始图像与标签的元组 `(image_uint8[28,28], label_int)`，**并非** DGLGraph。Loader 读取时会将图像重建成规则网格图。
  - **特征说明:** 原始存储中无图相关特征字段。

## 7. `molhiv`
- **基本信息**
  - **主题:** 分子 HIV 活性分类
  - **Metric:** ROC-AUC
  - **来源:** OGB `ogbg-molhiv`
  - **参考处理脚本:** `data/molhiv/preprocess_molhiv.py`
  - **样本规模:** 41,127
- **文件与入口**
  - **主数据文件:** `data/molhiv/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("molhiv", cfg).load_data()`
  - **重建入口:** `src.data.loader.molhiv_loader.MOLHIVLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 会把 label 包装为 `properties.label`)
  - **节点特征 (ndata):** `['feat', 'node_token_ids', 'node_type_id']`
    - 细节: `feat`: shape=(19, 1), int64 | `node_token_ids`: shape=(19, 1), int64 | `node_type_id`: shape=(19,), int64
  - **边特征 (edata):** `['edge_token_ids', 'edge_type_id', 'feat']`
    - 细节: `edge_token_ids`: shape=(40, 1), int64 | `edge_type_id`: shape=(40,), int64 | `feat`: shape=(40, 1), int64

## 8. `mutagenicity`
- **基本信息**
  - **主题:** 分子致突变性分类
  - **Metric:** Accuracy
  - **来源:** TU `Mutagenicity`
  - **参考处理脚本:** `data/mutagenicity/preprocess_mutagenicity.py`
  - **样本规模:** 4,337
- **文件与入口**
  - **主数据文件:** `data/mutagenicity/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("mutagenicity", cfg).load_data()`
  - **重建入口:** `src.data.loader.mutagenicity_loader.MutagenicityLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(16,), int64 | `node_labels`: shape=(16, 1), int64 | `node_token_ids`: shape=(16, 1), int64 | `node_type_id`: shape=(16,), int64
  - **边特征 (edata):** `['_ID', 'edge_labels', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(32,), int64 | `edge_labels`: shape=(32, 1), int64 | `edge_token_ids`: shape=(32, 1), int64 | `edge_type_id`: shape=(32,), int64

## 9. `peptides_func`
- **基本信息**
  - **主题:** 肽功能多标签分类
  - **Metric:** AP
  - **来源:** LRGB `Peptides-func`
  - **参考处理脚本:** `data/peptides_func/prepare_lrgb_data.py`
  - **样本规模:** 15,535
- **文件与入口**
  - **主数据文件:** `data/peptides_func/data.pkl.gz` *(注意是 gz 压缩格式)*
  - **相关侧文件:** `token_mappings.json`, `lrgb_analysis_results.json`
  - **统一 Loader 入口:** `get_dataloader("peptides_func", cfg).load_data()`
  - **重建入口:** `src.data.loader.peptides_func_loader.reconstruct_dgl_graph_from_lightweight`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[dict, numpy.ndarray]]` (单样本: `tuple[dict, numpy.ndarray]`)
  - **Label 类型:** `numpy.ndarray` (细节: shape=(10,), dtype=float32, preview=[0.0, 1.0, 0.0...])
  - **图载荷类型:** `lightweight_graph_dict`
  - **存储说明:** 当前仓库主文件保存的是**轻量图字典**，不直接存 DGLGraph；loader 会在读取时重建。
  - **原始字典图字段:** `['edge_features', 'edge_token_ids', 'edge_type_ids', 'edges', 'node_features', 'node_token_ids', 'node_type_ids', 'num_edges', 'num_nodes']`
    - 细节: `num_nodes` (int), `num_edges` (int), `edges` (tuple, len=2), `node_features` (shape=(338, 9), float32), `edge_features` (shape=(682, 3), float32), `node_token_ids` (shape=(338, 1), int32), `edge_token_ids` (shape=(682, 1), int32), `node_type_ids` (shape=(338,), int32), `edge_type_ids` (shape=(682,), int32)
  - **重建后特征映射:** - `ndata` -> `['x', 'node_token_ids', 'node_type_id']`
    - `edata` -> `['edge_attr', 'edge_token_ids', 'edge_type_id']`
- **直接读取示例:**
  ```python
  import gzip, pickle
  with gzip.open('data/peptides_func/data.pkl.gz', 'rb') as f:
      data = pickle.load(f)
  ```

## 10. `peptides_struct`
- **基本信息**
  - **主题:** 肽结构多目标回归
  - **Metric:** MAE
  - **来源:** LRGB `Peptides-struct`
  - **参考处理脚本:** `data/peptides_struct/prepare_lrgb_data.py`
  - **样本规模:** 15,535
- **文件与入口**
  - **主数据文件:** `data/peptides_struct/data.pkl.gz`
  - **相关侧文件:** `token_mappings.json`, `lrgb_analysis_results.json`
  - **统一 Loader 入口:** `get_dataloader("peptides_struct", cfg).load_data()`
  - **重建入口:** `src.data.loader.peptides_struct_loader.reconstruct_dgl_graph_from_lightweight`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[dict, numpy.ndarray]]` (单样本: `tuple[dict, numpy.ndarray]`)
  - **Label 类型:** `numpy.ndarray` (细节: shape=(11,), dtype=float32)
  - **图载荷类型:** `lightweight_graph_dict`
  - **存储说明:** 数据结构、字典字段结构及重建映射逻辑均 **同上方的 `peptides_func` 完全一致**。

## 11. `proteins`
- **基本信息**
  - **主题:** 蛋白质图分类
  - **Metric:** Accuracy
  - **来源:** TU `PROTEINS`
  - **参考处理脚本:** `data/proteins/preprocess_proteins.py`
  - **样本规模:** 1,113
- **文件与入口**
  - **主数据文件:** `data/proteins/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("proteins", cfg).load_data()`
  - **重建入口:** `src.data.loader.proteins_loader.PROTEINSLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_attr', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(42,), int64 | `node_attr`: shape=(42, 1), float64 | `node_labels`: shape=(42, 1), int64 | `node_token_ids`: shape=(42, 1), int64 | `node_type_id`: shape=(42,), int64
  - **边特征 (edata):** `['_ID', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(162,), int64 | `edge_token_ids`: shape=(162, 1), int64 | `edge_type_id`: shape=(162,), int64

## 12. `qm9`
- **基本信息**
  - **主题:** 分子多目标回归
  - **Metric:** MAE
  - **来源:** MoleculeNet / DGL built-in
  - **参考处理脚本:** `data/qm9/prepare_qm9_raw.py`
  - **样本规模:** 130,831
- **文件与入口**
  - **主数据文件:** `data/qm9/data.pkl`
  - **相关侧文件:** `smiles_1_direct.txt`, `smiles_2_explicit_h.txt`, `smiles_3_addhs.txt`, `smiles_4_addhs_explicit_h.txt`
  - **统一 Loader 入口:** `get_dataloader("qm9", cfg).load_data()`
  - **重建入口:** `src.data.loader.qm9_loader.QM9Loader._load_data_file`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, dict]]` (单样本: `tuple[DGLGraph, dict]`)
  - **Label 类型:** `dict` (包含键 `['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0']`)
    - Label 全集键值: `['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom']`
  - **图载荷类型:** `DGLGraph`
  - **存储说明:** 核心存储对象为 DGLGraph。Loader 读取时会再拼接四套 SMILES 并按 split 返回。
  - **节点特征 (ndata):** `['attr', 'pos']`
    - 细节: `pos`: shape=(14, 3), float32 | `attr`: shape=(14, 11), float32
  - **边特征 (edata):** `['edge_attr']`
    - 细节: `edge_attr`: shape=(32, 4), float32

## 13. `qm9test`
- **基本信息**
  - **主题:** QM9 子集多目标回归
  - **Metric:** MAE
  - **来源:** 派生自 `qm9`
  - **参考处理脚本:** `data/qm9test/create_qm9test_dataset.py`
  - **样本规模:** 13,083
- **文件与入口**
  - **主数据文件:** `data/qm9test/data.pkl`
  - **相关侧文件:** 同 `qm9`，额外新增 `metadata.json`
  - **统一 Loader 入口:** `get_dataloader("qm9test", cfg).load_data()`
  - **重建入口:** 继承流 `QM9TestLoader -> src.data.loader.qm9_loader.QM9Loader._load_data_file`
- **数据结构与特征**
  - 结构、特征维度、Label 细节均 **同上方的 `qm9` 完全一致**。Loader 继承 QM9 的读取逻辑，并额外读取 `metadata.json`。

## 14. `synthetic`
- **基本信息**
  - **主题:** 合成图分类
  - **Metric:** Accuracy
  - **来源:** TU `SYNTHETIC`
  - **参考处理脚本:** `data/synthetic/preprocess_synthetic.py`
  - **样本规模:** 300
- **文件与入口**
  - **主数据文件:** `data/synthetic/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("synthetic", cfg).load_data()`
  - **重建入口:** `src.data.loader.synthetic_loader.SYNTHETICLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_attr', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(100,), int64 | `node_attr`: shape=(100, 1), float64 | `node_labels`: shape=(100, 1), int64 | `node_token_ids`: shape=(100, 1), int64 | `node_type_id`: shape=(100,), int64
  - **边特征 (edata):** `['_ID', 'edge_token_ids', 'edge_type_id']`
    - 细节: `_ID`: shape=(392,), int64 | `edge_token_ids`: shape=(392, 1), int64 | `edge_type_id`: shape=(392,), int64

## 15. `twitter`
- **基本信息**
  - **主题:** 社交图分类
  - **Metric:** Accuracy
  - **来源:** TU `TWITTER-Real-Graph-Partial`
  - **参考处理脚本:** `data/twitter/preprocess_twitter_real_graph_partial.py`
  - **样本规模:** 144,033
- **文件与入口**
  - **主数据文件:** `data/twitter/data.pkl`
  - **统一 Loader 入口:** `get_dataloader("twitter", cfg).load_data()`
  - **重建入口:** `src.data.loader.twitter_loader.TwitterLoader._load_processed_data`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, int]]` (单样本: `tuple[DGLGraph, int]`)
  - **Label 类型:** `int` (`value_type=int`)
  - **图载荷类型:** `DGLGraph` (存储图对象即 DGLGraph；loader 直接按索引切分)
  - **节点特征 (ndata):** `['_ID', 'node_labels', 'node_token_ids', 'node_type_id']`
    - 细节: `_ID`: shape=(4,), int64 | `node_labels`: shape=(4, 1), int64 | `node_token_ids`: shape=(4, 1), int64 | `node_type_id`: shape=(4,), int64
  - **边特征 (edata):** `['_ID', 'edge_token_ids', 'edge_type_id', 'node_labels']`
    - 细节: `_ID`: shape=(6,), int64 | `node_labels`: shape=(6, 1), float64 | `edge_token_ids`: shape=(6, 1), int64 | `edge_type_id`: shape=(6,), int64

## 16. `zinc`
- **基本信息**
  - **主题:** 分子回归
  - **Metric:** MAE
  - **来源:** benchmarking-gnns `ZINC.pkl`
  - **参考处理脚本:** `data/zinc/prepare_zinc_raw.py`
  - **样本规模:** 12,000
- **文件与入口**
  - **主数据文件:** `data/zinc/data.pkl`
  - **相关侧文件:** `smiles_1_direct.txt`, `smiles_2_explicit_h.txt`, `smiles_3_addhs.txt`, `smiles_4_addhs_explicit_h.txt`, `conversion_stats.json`
  - **统一 Loader 入口:** `get_dataloader("zinc", cfg).load_data()`
  - **重建入口:** `src.data.loader.zinc_loader.ZINCLoader._load_data_file`
- **数据结构与特征**
  - **顶层结构:** `list[tuple[DGLGraph, torch.Tensor]]` (单样本: `tuple[DGLGraph, torch.Tensor]`)
  - **Label 类型:** `torch.Tensor` (细节: shape=(1,), dtype=torch.float32, preview=[0.8350355])
  - **图载荷类型:** `DGLGraph`
  - **存储说明:** 存储图对象即 DGLGraph；loader 读取时会把 tensor label 包成字典并额外加载四套 SMILES。
  - **节点特征 (ndata):** `['feat']`，细节: `{'feat': 'shape=(54,), dtype=torch.int64'}`
  - **边特征 (edata):** `['feat']`，细节: `{'feat': 'shape=(114,), dtype=torch.int64'}`