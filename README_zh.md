# TokenizerGraph

[[English README]](README.md)

> **分支说明：** 当前为 **`release`** 分支 — 仅包含复现论文实验所需的精简代码。
> 完整开发版本（工具脚本、benchmark、内部文档）请切换到 **[`dev` 分支](../../tree/dev)**。

TokenizerGraph 是一个基于图序列化的分子属性预测框架。它将分子图转换为 token 序列，通过 BPE 压缩发现子结构模式，再用 Transformer 编码器（BERT 或 GTE）进行预训练和下游任务微调。

## 概述

核心思路：将分子图序列化为线性 token 序列，然后像处理自然语言一样用语言模型学习。不同的序列化策略（Euler 路径、DFS、BFS 等）捕捉图的不同结构特征。BPE 压缩将高频 token 对合并为更高层的 token，相当于自动学习常见子结构。

```
原始分子 → 图构建 → 序列化 → BPE 压缩 → Transformer → 预测
```

## 项目结构

```
tokenizerGraph/
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
│   │   ├── serializer/         # 图序列化算法（Euler、DFS、BFS、Topo、SMILES 等）
│   │   └── compression/        # BPE 引擎（C++ / Numba / Python 后端）
│   ├── data/                   # 数据加载与预处理
│   │   └── loader/             # 各数据集加载器
│   ├── models/                 # 模型定义
│   │   ├── bert/               # BERT 编码器
│   │   ├── gte/                # GTE 编码器
│   │   └── unified_encoder.py  # 统一编码器接口
│   ├── training/               # 训练流程
│   └── utils/                  # 日志、指标、可视化
├── gte_model/                  # GTE 模型本地配置
├── final/                      # 论文实验脚本与绘图代码
└── docs/                       # 文档
```

## 安装

```bash
git clone <repository_url>
cd TokenizerGraph

# 开发模式安装
pip install -e .

# 编译 C++ BPE 后端（可选，推荐以提升速度）
python setup.py build_ext --inplace
```

主要依赖：`torch`、`dgl`、`networkx`、`rdkit`、`transformers`、`pybind11`、`pandas`。

## 使用方法

### 1. 数据预处理

序列化分子图并训练 BPE 分词器：

```bash
python prepare_data_new.py \
    --datasets qm9test \
    --methods feuler \
    --bpe_merges 2000
```

该脚本会加载数据集，用指定方法序列化所有图，在生成的序列上训练 BPE 模型，并构建词表。所有中间产物会被缓存以供复用。

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

在下游属性预测任务上微调预训练模型：

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

- **主实验** — `final/exp1_main/run/`：预训练与微调命令
- **效率分析** — `final/exp1_speed/`：序列化速度、token 长度统计、训练吞吐量
- **多重采样对比** — `final/exp2_mult_seralize_comp/`：多次序列化采样的效果
- **BPE 词表可视化** — `final/exp4_bpe_vocab_visual/`：codebook 检查与可视化

## 文档

- [配置指南](docs/guides/config_guide.md) — 配置文件结构与参数说明
- [实验指南](docs/guides/experiment_guide.md) — 实验设计与执行方法
- [BPE 使用指南](docs/bpe/BPE_USAGE_GUIDE.md) — BPE 引擎 API 与使用方法

## 分支说明

- **`release`** — 精简版本，只保留复现论文实验所需的代码和文档。
- **`dev`** — 完整开发版本，包含所有工具脚本、benchmark 和内部文档。
