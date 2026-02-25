# TokenizerGraph

将图结构数据序列化为 token 序列，并通过 BPE 压缩与 Transformer 编码器（BERT/GTE）进行图级预训练与微调。

## 项目结构

```
tokenizerGraph/
├── config.py                        # 统一配置管理
├── config/default_config.yml        # 默认配置文件
├── prepare_data_new.py              # 数据预处理（序列化 + BPE）
├── run_pretrain.py                  # 预训练入口
├── run_finetune.py                  # 微调入口
├── batch_pretrain_simple.py         # 批量预训练（多数据集/方法/GPU）
├── batch_finetune_simple.py         # 批量微调
├── aggregate_results.py             # 结果聚合
├── src/
│   ├── algorithms/
│   │   ├── serializer/              # 图序列化算法（Euler、DFS、BFS、Topo等）
│   │   └── compression/             # BPE 压缩引擎（Python + C++ 后端）
│   ├── data/                        # 数据加载与预处理
│   │   └── loader/                  # 各数据集加载器
│   ├── models/                      # 模型定义
│   │   ├── bert/                    # BERT 编码器
│   │   ├── gte/                     # GTE 编码器
│   │   └── unified_encoder.py       # 统一编码器接口
│   ├── training/                    # 训练流程
│   └── utils/                       # 工具函数
├── gte_model/                       # GTE 预训练模型配置
├── final/                           # 论文实验脚本与绘图代码
└── docs/                            # 文档
```

## 快速开始

### 环境准备

```bash
pip install -e .
```

### 1. 数据预处理

```bash
python prepare_data_new.py \
    --dataset qm9test \
    --method feuler \
    --bpe_num_merges 2000
```

### 2. 预训练

```bash
python run_pretrain.py \
    --dataset qm9test \
    --method feuler \
    --epochs 100 \
    --batch_size 256
```

### 3. 微调

```bash
python run_finetune.py \
    --dataset qm9test \
    --method feuler \
    --task regression \
    --target_property homo
```

### 4. 批量实验

```bash
python batch_pretrain_simple.py \
    --datasets qm9,zinc \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1

python batch_finetune_simple.py \
    --datasets qm9,zinc \
    --methods feuler,eulerian,cpp \
    --bpe_scenarios all,raw \
    --gpus 0,1
```

## 复现论文实验

论文中各实验的运行脚本位于 `final/` 目录：

- **主实验**: `final/exp1_main/run/` — 预训练与微调脚本
- **效率分析**: `final/exp1_speed/` — 序列化速度、token 长度、训练效率
- **多重采样对比**: `final/exp2_mult_seralize_comp/` — 多重序列化效果对比
- **BPE 词表可视化**: `final/exp4_bpe_vocab_visual/` — 词表结构分析与可视化

## 文档

- **[配置指南](docs/guides/config_guide.md)** — 配置文件结构与参数说明
- **[实验指南](docs/guides/experiment_guide.md)** — 实验设计与执行流程
- **[BPE 使用指南](docs/bpe/BPE_USAGE_GUIDE.md)** — BPE 压缩引擎使用方法

## 分支说明

- **`release`** — 最小复现版本，仅包含论文实验所需的代码与文档
- **`dev`** — 完整开发版本，保留所有原始工具脚本与文档
