# Cold-Start Runbook

## Objective

在独立克隆目录中，从公开原始数据开始执行一次冷启动复现，并将生成结果与当前仓库中的既有处理后数据进行结构对比。

## Candidate Datasets

### Primary target: `molhiv`

理由：

1. 当前仓库已有明确脚本：`data/molhiv/preprocess_molhiv.py`
2. 公开来源明确：OGB `ogbg-molhiv`
3. 依赖已在当前环境中可见：`ogb`、`torch_geometric`、`dgl`
4. 结果格式与当前 loader 直接对齐

### Fallback target: `mnist_raw`

理由：

1. 当前仓库已有明确脚本：`data/mnist_raw/prepare.py`
2. 公开来源明确：`torchvision.datasets.MNIST`
3. 下载与预处理链条更短

## Runtime Assumptions

- CPU cores: `128`
- Available memory: about `202 GiB`
- Disk:
  - `/tmp`: enough for a fresh clone and generated data
  - current workspace filesystem: enough for comparison reads
- GPU status: preprocessing阶段不依赖 GPU，本轮不要求训练入口通过

## Planned Steps

1. 在 `/tmp` 创建独立克隆目录。
2. 使用仓库远程地址克隆：
   - `git@github.com:BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers.git`
3. 在独立目录中执行目标数据集的公开来源预处理脚本。
4. 将新生成的 `data/<dataset>` 与当前仓库既有 `data/<dataset>` 对比：
   - 文件名集合
   - split 长度
   - `data.pkl` 样本数
   - 图对象字段结构
   - 标签字段结构
5. 记录结果与差异。

## Actual Run Status on 2026-03-15

### Remote clone result

The repository was cloned successfully into:

- `/tmp/tokenizerGraph-cold-start`

The remote default branch was:

- `release`

### Blocking observation

Immediately after clone:

- `data/` only contained `DATASET_STATS.md`
- dataset preprocessing scripts such as `data/molhiv/preprocess_molhiv.py` and `data/mnist_raw/prepare.py` were absent

The clone was then switched to:

- `dev`

The same result remained true:

- `data/molhiv/preprocess_molhiv.py` absent
- `data/mnist_raw/prepare.py` absent

### Interpretation

This means the current remote repository does not yet include the local dataset preprocessing scripts that exist in the present working tree. Therefore, remote cold-start reproduction is blocked before any dataset-specific command can be executed.

### Next required action before rerun

1. Track the local preprocessing scripts in git.
2. Normalize which of them are official cold-start entry points.
3. Repeat the independent clone-based run after those scripts are part of the repository history.

## Success Criteria

本轮冷启动复现成功，至少需要满足：

1. 独立克隆目录中能从公开来源下载并生成 `data/<dataset>`。
2. 生成结果可被当前 loader 读取。
3. 与当前仓库的既有处理后数据在结构上保持一致。

## Failure Recording Rule

若任一步失败，必须记录：

1. 失败命令
2. 失败阶段
3. 外部依赖还是仓库逻辑导致
4. 是否可通过文档或脚本修复
