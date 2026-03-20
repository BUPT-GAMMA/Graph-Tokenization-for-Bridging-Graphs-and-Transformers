# Reproducibility Environment Setup

## Scope

本文件只解决“如何准备一个能运行本仓库冷启动脚本与最小 smoke test 的环境”，不替代数据集级别说明。

核心结论：

- `pip install -e .` 现在已经能自动补齐 `pybind11` 构建依赖并完成 C++ 扩展构建。
- 但它不会自动安装运行时依赖；`torch`、`dgl`、`torch_geometric`、`transformers`、`rdkit` 等仍需要你先准备好。
- `run_finetune.py` 当前仍会在启动时要求 CUDA；没有可见 GPU 的环境不能宣称已经通过完整训练复现。

## Tested Baseline

当前仓库文档与测试主要基于如下环境快照整理：

- Python `3.10`
- PyTorch `2.1.2+cu121`
- DGL `2.4.0+cu121`
- PyG `2.4.0`
- Transformers `4.47.1`
- RDKit `2025.3.5` / `rdkit-pypi 2022.9.5`

完整历史环境快照保存在仓库根目录的 [`env.txt`](../../env.txt)。

## Recommended Setup Order

建议先创建独立环境，再按“底层框架 -> 图学习栈 -> 其余 Python 依赖 -> 本仓库”的顺序安装。

示例流程：

```bash
conda create -n pthgnn python=3.10 -y
conda activate pthgnn

# 1. 安装与你机器匹配的 PyTorch / CUDA 版本
# 2. 安装与该 PyTorch/CUDA 版本匹配的 DGL
# 3. 安装与该 PyTorch/CUDA 版本匹配的 torch-geometric 及其扩展

# 再安装通用依赖
pip install transformers pandas networkx ogb pybind11 outdated sentence-transformers

# RDKit 可按你的平台选择 conda 或 pip 方案

# 最后安装当前仓库
pip install -e .
```

## What Was Verified In This Round

本轮已新鲜验证：

- 在全新虚拟环境中执行 `pip install -e /home/gzy/py/tokenizerGraph`，构建依赖安装和 editable C++ 扩展构建成功。
- 文档/脚本/`qm9` 血缘相关的最小测试集通过：
  - `tests/test_reproducibility_documentation.py`
  - `tests/test_data_preprocess_script_inventory.py`
  - `tests/test_qm9_raw_script_scaffold.py`
  - `tests/test_qm9_lineage.py`
  - `tests/test_repro_compare.py`

本轮未能在当前会话中宣称通过的事项：

- `torch.cuda.is_available()` 在当前会话中为 `False`
- `nvidia-smi` 返回 `Failed to initialize NVML: Unknown Error`
- 因此 `run_finetune.py` 的 CUDA 路径不能在本会话里被视为已验证通过

## Recommended Verification Commands

环境准备完成后，先跑最小验证：

```bash
pytest tests/test_reproducibility_documentation.py \
       tests/test_data_preprocess_script_inventory.py \
       tests/test_qm9_raw_script_scaffold.py \
       tests/test_qm9_lineage.py \
       tests/test_repro_compare.py -v
```

随后再进入数据与训练链路：

```bash
python prepare_data_new.py --datasets qm9test --methods feuler --bpe_merges 2000
python run_pretrain.py --dataset qm9test --method feuler --experiment_group smoke
python run_finetune.py --dataset qm9test --method feuler --experiment_group smoke
```

## Failure Interpretation

- 如果 `pip install -e .` 失败在 `pybind11`，说明构建依赖没有从索引成功安装，优先检查网络或包源。
- 如果安装完成后导入时报 `ModuleNotFoundError: torch`，说明运行时依赖环境还没补齐，这不是 C++ 扩展构建失败。
- 如果训练脚本失败在 CUDA 检查，说明当前会话没有可见 GPU；这属于环境前置条件未满足，不是数据准备脚本本身失败。
