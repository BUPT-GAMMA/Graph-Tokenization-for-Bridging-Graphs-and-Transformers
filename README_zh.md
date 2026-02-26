# GraphTokenizer

**Graph Tokenization for Bridging Graphs and Transformers**

[[English README]](README.md) · [[论文 (ICLR 2026)]](https://openreview.net/forum?id=PLACEHOLDER)

> **分支说明：** `release` — 复现论文实验的精简代码。[`dev`](../../tree/dev) — 包含工具脚本、benchmark 和内部文档的完整开发版本。

## 概述

GraphTokenizer 是一个通用的**图分词**框架，将任意带标签的图转换为离散 token 序列，使标准的现成 Transformer 模型（如 BERT、GTE）能够**无需任何架构修改**即可直接处理图结构数据。

该框架将**可逆图序列化**与**字节对编码（BPE）** 相结合。结构引导的序列化机制利用图子结构的全局统计信息，确保高频子结构在序列中表现为相邻符号——这正是 BPE 学习有意义的结构图 token 词表的理想输入。

<p align="center">
  <img src="docs/assets/framework.jpg" width="90%" alt="GraphTokenizer Framework">
</p>

**框架概览。** (A) 从训练图中收集子结构频率统计。(B) 通过频率引导的欧拉回路进行结构引导的可逆序列化。(C) 在序列化语料上训练 BPE 词表，将图编码为离散 token 供下游序列模型使用。

```
带标签的图 → 结构引导的序列化 → BPE 分词 → Transformer → 预测
```

### 核心贡献

- **通用图分词框架。** 将可逆序列化与 BPE 结合，构建图与序列模型之间的接口，使标准 Transformer 无需修改即可处理图数据。
- **面向 BPE 的结构引导序列化。** 基于全局子结构统计的确定性序列化方法，解决图的排列歧义，将高频子结构对齐为相邻模式以供 BPE 合并。
- **14 个基准数据集上达到最优。** 在涵盖分子、生物医学、社交网络、学术网络和合成图等多领域的图分类与回归基准上取得 SOTA 结果。

### 主要结果

| 模型 | molhiv (AUC↑) | p-func (AP↑) | mutag (Acc↑) | coildel (Acc↑) | dblp (Acc↑) | qm9 (MAE↓) | zinc (MAE↓) | aqsol (MAE↓) | p-struct (MAE↓) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GCN | 74.0 | 53.2 | 79.7 | 74.6 | 76.6 | 0.134 | 0.399 | 1.345 | 0.342 |
| GIN | 76.1 | 61.4 | 80.4 | 72.0 | 73.8 | 0.176 | 0.379 | 2.053 | 0.338 |
| GatedGCN | 80.6 | 51.2 | 83.6 | 83.7 | 86.0 | 0.096 | 0.370 | 0.940 | 0.312 |
| GraphGPS | 78.5 | 53.5 | 84.3 | 80.5 | 71.6 | 0.084 | 0.310 | 1.587 | 0.251 |
| Exphormer | 82.3 | 64.5 | 82.7 | **91.5** | 84.9 | 0.080 | 0.281 | 0.749 | 0.251 |
| GraphMamba | 81.2 | 67.7 | 85.0 | 74.5 | 87.6 | 0.083 | 0.209 | 1.133 | 0.248 |
| GCN+ | 80.1 | 72.6 | 88.7 | 88.9 | 89.6 | 0.077 | **0.116** | 0.712 | 0.244 |
| **GT+BERT** | 82.6 | 68.5 | 87.5 | 74.1 | 93.2 | 0.122 | 0.241 | 0.648 | 0.247 |
| **GT+GTE** | **87.4** | **73.1** | **90.1** | 89.6 | **93.6** | **0.071** | 0.131 | **0.609** | **0.242** |

结果为 5 次独立运行的均值。加粗 = 最优。完整 14 个数据集的结果请参见论文。

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

# 开发模式安装
pip install -e .

# 编译 C++ BPE 后端（可选，推荐以提升速度）
python setup.py build_ext --inplace
```

主要依赖：`torch`、`dgl`、`networkx`、`rdkit`、`transformers`、`pybind11`、`pandas`。

## 快速开始

### 1. 数据预处理

序列化图并训练 BPE 分词器：

```bash
python prepare_data_new.py \
    --datasets qm9test \
    --methods feuler \
    --bpe_merges 2000
```

该脚本加载数据集，用指定方法（如频率引导欧拉回路）序列化所有图，在生成的序列上训练 BPE 模型，并构建词表。所有中间产物会被缓存以供复用。

### 2. 预训练

用掩码语言模型（MLM）任务预训练 Transformer 编码器：

```bash
python run_pretrain.py \
    --dataset qm9test \
    --method feuler \
    --experiment_group my_experiment \
    --epochs 100 \
    --batch_size 256
```

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
- **BPE 词表可视化** — `final/exp4_bpe_vocab_visual/`：codebook 检查与可视化

## 文档

- [配置指南](docs/guides/config_guide.md) — 配置文件结构与参数说明
- [实验指南](docs/guides/experiment_guide.md) — 实验设计与执行方法
- [BPE 使用指南](docs/bpe/BPE_USAGE_GUIDE.md) — BPE 引擎 API 与使用方法

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
