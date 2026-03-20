# GraphTokenizer

**Graph Tokenization for Bridging Graphs and Transformers**

[[English README]](README.md) · [[论文 (ICLR 2026 / OpenReview)]](https://openreview.net/forum?id=jCctxI1BGF) · arXiv（即将发布）

> **分支说明：** `release` — 复现论文实验的精简代码。[`dev`](../../tree/dev) — 包含工具脚本、benchmark 和内部文档的完整开发版本。

## 概述

大规模预训练 Transformer 的成功与分词器密切相关，分词器将原始输入转换为离散符号。**GraphTokenizer** 将这一范式扩展到图结构数据，提出了一个通用的**图分词**框架。它将任意带标签的图转换为离散 token 序列，使标准的现成 Transformer 模型（如 BERT、GTE）能够**无需任何架构修改**即可直接处理图数据。

该框架将**可逆图序列化**与**字节对编码（BPE）** 相结合，后者是大语言模型中的事实标准分词器。为了更好地捕获结构信息，序列化过程由**图子结构的全局统计信息**引导，确保高频子结构在生成的序列中表现为相邻符号——这正是 BPE 发现有意义的结构图 token 词表的理想输入。整个过程是**可逆的**：原始图可以从其 token 序列中完整重建。

<p align="center">
  <img src="docs/assets/framework.jpg" width="90%" alt="GraphTokenizer Framework">
</p>

**框架概览。** **(A)** 从训练图中收集子结构频率统计（带标签的边模式）。**(B)** 通过频率引导的欧拉回路进行结构引导的可逆序列化——在每个节点处，根据频率优先级选择下一条边（例如，在红色 C 节点处，C–C 模式频率最高，因此优先遍历该边）。**(C)** 在序列化语料上训练 BPE 词表；BPE 迭代地将最高频的相邻符号对合并为新 token，将序列压缩至原始长度的约 10%，同时保留常见子结构。

```
带标签的图  →  结构引导的序列化  →  BPE 分词  →  Transformer  →  预测
```

### 核心贡献

- **通用图分词框架。** 将可逆图序列化与 BPE 结合，构建图与序列模型之间的双向接口。通过将图结构编码与模型架构解耦，使标准现成 Transformer 无需任何架构修改即可处理图数据。
- **面向 BPE 的结构引导序列化。** 基于全局子结构统计的确定性序列化机制，解决图固有的排列歧义（置换不变性），将高频子结构系统性地排列为相邻符号模式——恰好是 BPE 贪心合并策略所擅长利用的输入。
- **14 个基准数据集上达到最优。** 在涵盖分子、生物医学、社交网络、学术网络和合成图等多领域的图分类与回归基准上取得 SOTA 结果。从紧凑的 BERT-small 扩展到更大的 GTE 主干带来一致的性能提升，表明图分词可以利用 Transformer 已被验证的扩展规律。

### 主要结果

分类（↑ 越高越好）和回归（↓ 越低越好）结果：

| 模型 | molhiv (AUC↑) | p-func (AP↑) | mutag (Acc↑) | coildel (Acc↑) | dblp (Acc↑) | qm9 (MAE↓) | zinc (MAE↓) | aqsol (MAE↓) | p-struct (MAE↓) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCN | 74.0 | 53.2 | 79.7 | 74.6 | 76.6 | 0.134 | 0.399 | 1.345 | 0.342 |
| GIN | 76.1 | 61.4 | 80.4 | 72.0 | 73.8 | 0.176 | 0.379 | 2.053 | 0.338 |
| GAT | 72.1 | 51.2 | 80.1 | 74.4 | 76.3 | 0.114 | 0.445 | 1.388 | 0.316 |
| GatedGCN | 80.6 | 51.2 | 83.6 | 83.7 | 86.0 | 0.096 | 0.370 | 0.940 | 0.312 |
| GraphGPS | 78.5 | 53.5 | 84.3 | 80.5 | 71.6 | 0.084 | 0.310 | 1.587 | 0.251 |
| Exphormer | 82.3 | 64.5 | 82.7 | **91.5** | 84.9 | 0.080 | 0.281 | 0.749 | 0.251 |
| GraphMamba | 81.2 | 67.7 | 85.0 | 74.5 | 87.6 | 0.083 | 0.209 | 1.133 | 0.248 |
| GCN+ | 80.1 | 72.6 | 88.7 | 88.9 | 89.6 | 0.077 | **0.116** | 0.712 | 0.244 |
| **GT+BERT** | 82.6 | 68.5 | 87.5 | 74.1 | 93.2 | 0.122 | 0.241 | 0.648 | 0.247 |
| **GT+GTE** | **87.4** | **73.1** | **90.1** | 89.6 | **93.6** | **0.071** | 0.131 | **0.609** | **0.242** |

结果为 5 次独立运行的均值。**加粗** = 最优。完整 14 个数据集（含 DD、Twitter、Proteins、Colors-3、Synthetic）的结果请参见论文。

### 支持的序列化方法

| 方法 | 可逆 | 确定性 | 适用范围 |
|:---|:---:|:---:|:---|
| 频率引导欧拉回路 (Feuler) | ✅ | ✅ | 任意带标签图 |
| 频率引导中国邮路 (FCPP) | ✅ | ✅ | 任意带标签图 |
| 欧拉回路 | ✅ | ❌ | 任意带标签图 |
| 中国邮路问题 (CPP) | ✅ | ❌ | 任意带标签图 |
| 规范 SMILES | ✅ | ✅ | 仅限分子图 |
| DFS / BFS / Topo | ❌ | ❌ | 任意图 |

默认方法为 **Feuler**（频率引导欧拉回路），同时满足可逆性和确定性，时间复杂度为 O(|E|)。

## 项目结构

```
GraphTokenizer/
├── prepare_data_new.py         # 数据预处理：序列化 + BPE 训练 + 词表构建
├── run_pretrain.py             # 预训练入口（MLM）
├── run_finetune.py             # 微调入口（回归/分类）
├── batch_pretrain_simple.py    # 批量预训练（跨数据集/方法/GPU）
├── batch_finetune_simple.py    # 批量微调
├── aggregate_results.py        # 收集和汇总实验结果
├── config.py                   # 统一配置管理
├── config/default_config.yml   # 默认配置
├── src/
│   ├── algorithms/
│   │   ├── serializer/         # 图序列化算法（频率引导欧拉、Euler、DFS、BFS、Topo、SMILES、CPP 等）
│   │   └── compression/        # BPE 引擎（C++ / Numba / Python 后端）
│   ├── data/                   # 统一数据接口与各数据集加载器
│   │   └── loader/             # 各数据集加载器（QM9、ZINC、AQSOL、MNIST、Peptides 等）
│   ├── models/                 # 模型定义
│   │   ├── bert/               # BERT 编码器、词表管理、数据流水线
│   │   ├── gte/                # GTE 编码器（Alibaba-NLP/gte-multilingual-base）
│   │   └── unified_encoder.py  # 统一编码器接口
│   ├── training/               # 训练流程（预训练、微调、评估）
│   └── utils/                  # 日志、指标、可视化
├── gte_model/                  # GTE 模型本地配置（离线使用）
├── final/                      # 论文实验脚本与绘图代码
└── docs/                       # 文档
```

## 安装

```bash
git clone https://github.com/BUPT-GAMMA/GraphTokenizer.git
cd GraphTokenizer

# 开发模式安装。
# 当前仓库中的 pyproject.toml 已声明 pybind11 构建依赖，
# 在可联网环境中 pip 会自动补齐 C++ 扩展构建所需依赖。
pip install -e .

# 编译 C++ BPE 后端（可选，推荐以提升速度）
python setup.py build_ext --inplace
```

主要依赖：`torch`、`dgl`、`networkx`、`rdkit`、`transformers`、`pybind11`、`pandas`。

说明：

- 如果是在离线环境安装，请先在目标环境中预装 `pybind11`，再执行 `pip install -e .`。
- `pip install -e .` 只会处理本地包及其构建依赖；真正运行实验仍需要环境里已有 `torch`、`dgl`、`rdkit`、`transformers` 等运行时依赖。

## 快速开始

### 1. 数据预处理

在运行 `prepare_data_new.py` 之前，请先确认 `data/<dataset>/` 下已经放好了数据集原始/预处理文件。

当前各 loader 默认假设目录中至少存在：

```text
data/<dataset>/
├── data.pkl
├── train_index.json
├── val_index.json
└── test_index.json
```

对于 `qm9`、`zinc` 这类分子数据集，部分 loader 还会尝试读取可选的 SMILES 文件，例如 `smiles_1_direct.txt`。

对于干净克隆，`qm9test` 不应视为仓库内已带好的示例数据，而应视为一个派生出来的 smoke-test 数据集：先从公开来源构建 `qm9`，再通过 `data/qm9test/create_qm9test_dataset.py` 生成 `qm9test`。

序列化图并训练 BPE 分词器：

```bash
python prepare_data_new.py \
    --datasets qm9test \
    --methods feuler \
    --bpe_merges 2000
```

该脚本会：

- 读取 `data/qm9test/data.pkl` 及固定划分文件
- 用指定方法对所有图做序列化
- 在序列化语料上训练 BPE
- 构建下游 Transformer 所需词表
- 将缓存产物写入 `data/processed/<dataset>/...`

执行完成后，通常会生成类似下面的文件：

```text
data/processed/qm9test/
├── serialized_data/feuler/single/serialized_data.pickle
└── vocab/feuler/bpe/single/vocab.json
```

正式的 release 版数据准备与执行说明见：

- [`src/data/README.md`](src/data/README.md) — 数据层接口约定与目录结构
- [`docs/reproducibility/environment-setup.md`](docs/reproducibility/environment-setup.md) — 已验证环境边界、依赖分层与安装验证说明
- [`docs/reproducibility/paper-dataset-cold-start-guide.md`](docs/reproducibility/paper-dataset-cold-start-guide.md) — 论文范围内数据集的正式准备与验证指南

当前 release 口径：

- `release` 只保留论文范围内的正式复现入口及其最小验证资产
- 实验性、受阻或审计型脚本统一留在 `dev` 分支
- `qm9test` 是从 `qm9` 派生的 smoke-test 数据集，干净克隆里需要先生成 `qm9` 再生成它

执行说明：

- `prepare_data_new.py` 使用复数参数 `--datasets`、`--methods`，`run_pretrain.py` 与 `run_finetune.py` 使用单数参数 `--dataset`、`--method`
- 使用 `--multiple_samples K` 生成数据时，训练阶段必须使用匹配的 `serialization.multiple_sampling.enabled=true` 与 `serialization.multiple_sampling.num_realizations=K`
- 默认配置中 `encoder.type: gte`，未显式切换时实际运行路径为 GTE 编码器

### 2. 预训练

使用 Masked Language Modeling (MLM) 预训练 Transformer 编码器：

```bash
python run_pretrain.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --epochs 100 \
    --batch_size 256
```

预训练说明：

- 这里必须使用单数参数 `--dataset` 和 `--method`
- 该脚本直接读取 `prepare_data_new.py` 生成的缓存结果
- 默认路径配置来自 `config/default_config.yml`，其中 `data_dir` 会解析到项目根目录下的 `data/`
- 更完整的实验性与审计型命令只保留在 `dev` 分支

### 3. 微调

在下游图预测任务上微调预训练模型：

```bash
python run_finetune.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --target_property homo \
    --epochs 200 \
    --batch_size 64
```

对于 `qm9` 这类回归数据集，建议显式指定 `--target_property`；对于 `mutagenicity`、`molhiv` 这类分类数据集，通常可以直接依赖 loader 内部元信息。

微调说明：

- `run_finetune.py` 启动时要求 CUDA 可用
- 最小验证应显式传入 `--pretrained_dir model/<group>/<exp_name>/run_0/best`
- 预训练目录中必须存在 `config.bin` 与 `pytorch_model.bin`

### 4. 批量实验

跨数据集、序列化方法和 GPU 并行运行实验：

```bash
python batch_pretrain_simple.py \
    --datasets qm9,zinc,mutagenicity \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1

python batch_finetune_simple.py \
    --datasets qm9,zinc,mutagenicity \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1
```

## 复现论文实验

论文各实验的运行脚本在 `final/` 目录下：

- **主实验** — `final/exp1_main/run/`：全部 14 个数据集的预训练与微调命令
- **效率分析** — `final/exp1_speed/`：序列化速度、token 长度统计、训练吞吐量
- **多重采样对比** — `final/exp2_mult_seralize_comp/`：多次序列化采样的效果
- **BPE 词表可视化** — `final/exp4_bpe_vocab_visual/`：码本分析与可视化

## 数据准备检查清单

建议按以下顺序检查数据集是否具备端到端运行条件：

1. 将数据放到 `data/<dataset>/`
2. 确认 `data.pkl`、`train_index.json`、`val_index.json`、`test_index.json` 全部存在
3. 确认该数据集名称已经在 `src/data/unified_data_factory.py` 中注册
4. 运行：
```bash
python prepare_data_new.py --datasets <dataset> --methods feuler
```

5. 检查 `data/processed/<dataset>/serialized_data/...` 与 `data/processed/<dataset>/vocab/...` 是否生成成功
6. 如果使用了多重采样，再确认生成目录到底是 `single/` 还是 `multi_<K>/`
7. 做一个最小预训练冒烟测试：

```bash
python run_pretrain.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
```

8. 再做一个最小微调冒烟测试：

```bash
python run_finetune.py --dataset <dataset> --method feuler --epochs 1 --batch_size 8
```

微调要求可用 CUDA 设备，并要求预训练目录中存在有效权重文件。

## 文档

- [配置指南](docs/guides/config_guide.md) — 配置文件结构与参数说明
- [实验指南](docs/guides/experiment_guide.md) — 如何设计与运行实验
- [BPE 使用指南](docs/bpe/BPE_USAGE_GUIDE.md) — BPE 引擎 API 与使用方法
- [环境准备说明](docs/reproducibility/environment-setup.md) — 已验证环境边界与安装说明
- [论文范围冷启动指南](docs/reproducibility/paper-dataset-cold-start-guide.md) — release 正式范围的数据准备入口

## 引用

如果本工作对您有帮助，请引用我们的论文：

```bibtex
@inproceedings{guo2026graphtokenizer,
  title={Graph Tokenization for Bridging Graphs and Transformers},
  author={Guo, Zeyuan and Diao, Enmao and Yang, Cheng and Shi, Chuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## 分支说明

- **`release`** — 精简版本，只保留复现论文实验所需的代码和文档。
- **`dev`** — 完整开发版本，包含所有工具脚本、benchmark 和内部文档。
