# Paper Dataset Cold-Start Guide

## Scope

本指南只覆盖论文主结果实际涉及的数据集，并只给出这部分数据集的正式操作流程。

当前纳入正式保证范围的数据集：

- `mnist_raw`
- `molhiv`
- `proteins`
- `colors3`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dblp`
- `dd`
- `twitter`
- `peptides_func`
- `peptides_struct`
- `qm9`
- `qm9test`

当前明确不纳入本轮正式保证范围：

- `zinc`
- `aqsol`
- `mnist`

## What "Success" Means

对本指南覆盖的数据集，当前文档口径以以下标准认定“已验证可用”：

1. 存在仓库内冷启动脚本。
2. 已实际运行过公开来源下载或公开来源恢复流程。
3. 生成的 `train_index.json / val_index.json / test_index.json` 与当前基线一致。
4. 若存在额外 side files，例如 `smiles_*`，则这些文件与当前基线一致。
5. `data.pkl` 至少达到语义一致；若 pickle 字节不一致，文档会明确标注这是 pickle 序列化层差异，而不是样本语义差异。

## Environment

建议环境：

```bash
conda activate pthgnn
pip install -e .
```

当前仓库已经验证过的文档与测试入口：

```bash
pytest tests/test_reproducibility_documentation.py -v
pytest tests/test_data_preprocess_script_inventory.py -v
pytest tests/test_molecular_dataset_current_format.py -v
pytest tests/test_repro_compare.py -v
pytest tests/test_qm9_lineage.py tests/test_qm9_raw_script_scaffold.py -v
```

## Standard Validation Command

对于任意已生成的数据集目录，统一使用下面的命令做基线对比：

```bash
python scripts/compare_dataset_artifacts.py \
  --baseline data/<dataset> \
  --candidate /tmp/<dataset>-run \
  --files data.pkl train_index.json val_index.json test_index.json
```

解释：

- `match=true` 表示文件字节级一致
- `semantic_match=true` 表示 `data.pkl` 虽非字节级一致，但样本语义一致

## Dataset Commands

### `mnist_raw`

```bash
python data/mnist_raw/prepare.py
```

当前状态：

- `data.pkl` 与三份 split 已验证字节级一致

### `molhiv`

```bash
python data/molhiv/preprocess_molhiv.py
```

当前状态：

- split 字节级一致
- `data.pkl` 语义一致

### TU / Synthetic Family

适用数据集：

- `proteins`
- `colors3`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dblp`
- `dd`
- `twitter`

对应命令：

```bash
python data/proteins/preprocess_proteins.py
python data/colors3/preprocess_colors3.py
python data/synthetic/preprocess_synthetic.py
python data/mutagenicity/preprocess_mutagenicity.py
python data/coildel/preprocess_coil_del.py
python data/dblp/preprocess_dblp_v1.py
python data/dd/preprocess_dd.py
python data/twitter/preprocess_twitter_real_graph_partial.py
```

当前状态：

- 三份 split 已验证字节级一致
- `data.pkl` 已验证语义一致

补充：

- `colors3` 当前基线使用顺序切片 `80/10/10`，而不是随机划分

### LRGB

```bash
python data/peptides_func/prepare_lrgb_data.py
python data/peptides_struct/prepare_lrgb_data.py
```

当前状态：

- 三份 split 已验证字节级一致
- `data.pkl.gz` 已验证语义一致

### `qm9`

如当前运行时需要代理，先设置：

```bash
export http_proxy=http://local.nginx.show:7890
export https_proxy=http://local.nginx.show:7890
```

然后运行：

```bash
python data/qm9/prepare_qm9_raw.py \
  --split-source-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --reference-data-pkl /home/gzy/py/tokenizerGraph/data/qm9/data.pkl \
  --reference-smiles-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --output-dir /tmp/qm9-run
```

当前状态：

- split 字节级一致
- 四份 `smiles_*` 字节级一致
- `data.pkl` 语义一致

### `qm9test`

```bash
python data/qm9test/create_qm9test_dataset.py \
  --original-indices-path /home/gzy/py/tokenizerGraph/data/qm9test/metadata.json \
  --source-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --output-dir /tmp/qm9test-run
```

当前状态：

- split 字节级一致
- 四份 `smiles_*` 字节级一致
- `metadata.json` 字节级一致
- `data.pkl` 语义一致

## Datasets Not Covered In This Round

### `zinc`

- 当前仓库保留实验性脚本
- 但论文主结果不依赖它
- 本轮不作为正式已实现冷启动流程对外承诺

### `aqsol`

- 当前仓库保留实验性脚本
- 但论文主结果不依赖它
- 本轮不作为正式已实现冷启动流程对外承诺

### `mnist`

- 当前仍依赖未标准化的 `final_slic` 路径
- 目前不算正式闭环

## Recommended Closing Checklist

在准备结束本轮时，至少确认：

```bash
pytest tests/test_reproducibility_documentation.py \
  tests/test_data_preprocess_script_inventory.py \
  tests/test_molecular_dataset_current_format.py \
  tests/test_repro_compare.py \
  tests/test_qm9_lineage.py \
  tests/test_qm9_raw_script_scaffold.py \
  tests/test_molecule_raw_script_scaffold.py -v
```

然后人工检查：

1. 论文范围内的数据集是否都在本指南中出现。
2. `zinc` / `aqsol` / `mnist` 是否明确标记为当前不纳入正式保证范围。
3. 每个已纳入范围的数据集是否都给出了实际命令和验证口径。
