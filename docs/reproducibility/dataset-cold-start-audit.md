# Dataset Cold-Start Audit

## Scope

本文件审计的是“从公开原始数据冷启动构建 `data/<dataset>`”的可复现性，而不是“当前仓库中是否已经存在处理后数据文件”。

现有 `data/<dataset>` 中的处理后数据仅作为对照基准，用于验证新的冷启动方案生成结果是否与既有格式一致。现有处理后数据不应被原地改写。

## Audit Rule

数据集只有在满足以下条件时，才记为 `Cold-start reproducible`：

1. 公开原始数据来源明确。
2. 仓库内或上级 `/home/gzy/py` 范围内存在明确处理脚本。
3. 脚本输出目标与当前 loader 所需格式一致，至少可生成：
   - `data.pkl`
   - `train_index.json`
   - `val_index.json`
   - `test_index.json`
4. 若 loader 依赖额外 side files（如分子数据集的 SMILES 文本），其生成方式也明确。

## Summary

Branch-scope note:

- `release` should only expose canonical paper-scope cold-start entrypoints
- `dev` keeps the broader audit surface, including experimental drafts, blocked loaders, and secondary helper scripts

### Paper-scope datasets with validated cold-start pipelines

- `molhiv`
- `peptides_func`
- `peptides_struct`
- `colors3`
- `proteins`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dblp`
- `dd`
- `twitter`
- `mnist_raw`
- `qm9`
- `qm9test`

### Present in repository but not yet validated end-to-end in the current runtime

- `code2`

Verified clone-based statuses so far:

- `mnist_raw` -> byte-identical to baseline
- `molhiv` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `proteins` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `colors3` -> semantic-equivalent to baseline; split files byte-identical after split-policy fix, `data.pkl` pickle bytes differ
- `peptides_func` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl.gz` bytes differ
- `peptides_struct` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl.gz` bytes differ
- `synthetic` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `mutagenicity` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `coildel` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `dd` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `dblp` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `twitter` -> semantic-equivalent to baseline; split files byte-identical, `data.pkl` pickle bytes differ
- `qm9` -> semantic-equivalent to baseline; split files byte-identical, four `smiles_*` files byte-identical, `data.pkl` pickle bytes differ
- `qm9test` -> semantic-equivalent to baseline; split files byte-identical, four `smiles_*` files byte-identical, `metadata.json` byte-identical, `data.pkl` pickle bytes differ

### Explicitly out of current paper-facing guarantee scope

- `zinc`
- `aqsol`
- `mnist`

说明：

- `zinc` / `aqsol` 在当前仓库中已经有实验性冷启动脚本和内部验证记录
- 但由于论文主结果不包含这两个数据集，本轮不再将它们列为“正式已实现并保证”的数据集
- 文档只保留它们的实验性记录，不将其纳入本轮交付边界

### Paper-scope datasets that are semantically reproducible but not byte-identical

- `molhiv`
- `proteins`
- `colors3`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dd`
- `dblp`
- `twitter`
- `peptides_func`
- `peptides_struct`
- `qm9`
- `qm9test`

## Dataset Matrix

| Dataset | Public source | Script status | Script path | Current assessment | Main gap |
| --- | --- | --- | --- | --- | --- |
| `code2` | OGB `ogbg-code2` | Present in current repo | `data/code2/preprocess_code2.py` | Not yet validated in current runtime | Upstream OGB download hit `HTTP 502`; no final baseline-equivalence result recorded yet |
| `molhiv` | OGB `ogbg-molhiv` | Present in current repo | `data/molhiv/preprocess_molhiv.py` | Cold-start reproducible | Needs formal runbook entry in main docs |
| `peptides_func` | LRGB `Peptides-func` | Present in current repo | `data/peptides_func/prepare_lrgb_data.py` | Cold-start reproducible | Script is heavy; main docs do not expose it as official cold-start path |
| `peptides_struct` | LRGB `Peptides-struct` | Present in current repo | `data/peptides_struct/prepare_lrgb_data.py` | Cold-start reproducible | Same as above |
| `colors3` | TU `COLORS-3` | Present in current repo | `data/colors3/preprocess_colors3.py` | Cold-start reproducible | Split is locally generated; docs must state this explicitly |
| `proteins` | TU `PROTEINS` | Present in current repo | `data/proteins/preprocess_proteins.py` | Cold-start reproducible | Split is locally generated |
| `synthetic` | TU `SYNTHETIC` | Present in current repo | `data/synthetic/preprocess_synthetic.py` | Cold-start reproducible | Split is locally generated |
| `mutagenicity` | TU `Mutagenicity` | Present in current repo | `data/mutagenicity/preprocess_mutagenicity.py` | Cold-start reproducible | Split is locally generated |
| `coildel` | TU `COIL-DEL` | Present in current repo | `data/coildel/preprocess_coil_del.py` | Cold-start reproducible | Split is locally generated |
| `dblp` | TU `DBLP_v1` | Present in current repo | `data/dblp/preprocess_dblp_v1.py` | Cold-start reproducible | Split is locally generated |
| `dd` | TU `DD` | Present in current repo | `data/dd/preprocess_dd.py` | Cold-start reproducible | Split is locally generated |
| `twitter` | TU `TWITTER-Real-Graph-Partial` | Present in current repo | `data/twitter/preprocess_twitter_real_graph_partial.py` | Cold-start reproducible | Public source should be surfaced more clearly in docs |
| `mnist_raw` | `torchvision.datasets.MNIST` | Present in current repo | `data/mnist_raw/prepare.py` | Cold-start reproducible | Needs formal cold-start runbook |
| `mnist` | `tensorflow.keras.datasets.mnist` | Present but incomplete | `data/mnist/convert_mnist_to_dgl.py` | Partially traceable | Depends on local `final_slic` pipeline and is not packaged as a clean reproducible cold-start path |
| `qm9` | MoleculeNet / DGL built-in | Raw scaffold present in current repo; alignment now verified against current baseline | `data/qm9/prepare_qm9_raw.py`, `data/qm9/process_qm9_dataset.py`, `/home/gzy/py/backup_tokenizerGraph/backup/legacy_scripts/qm9_loader.py` | Cold-start reproducible to semantic baseline equivalence | Public raw-source access still depends on runtime transport/proxy conditions; `data.pkl` bytes still differ |
| `qm9test` | Derived from `qm9` | Secondary replay script present in current repo | `data/qm9test/create_qm9test_dataset.py` | Reproducible secondary replay pipeline | Not a public-raw dataset by itself; `data.pkl` bytes still differ |
| `zinc` | benchmarking-gnns public `ZINC.pkl` | Experimental script present in current repo | `data/zinc/prepare_zinc_raw.py` | Out of current formal scope | Paper results do not depend on this dataset; current script remains an experimental draft |
| `aqsol` | AqSol raw zip + dictionary pipeline | Experimental script present in current repo | `data/aqsol/prepare_aqsol_raw.py` | Out of current formal scope | Paper results do not depend on this dataset; current script remains an experimental draft |

## Important Findings

### 0. The remote repository cannot currently reproduce the local script surface

An independent clone was created at `/tmp/tokenizerGraph-cold-start`.

Observed results:

- default remote branch: `release`
- after clone, `data/` only contained `DATASET_STATS.md`
- no dataset preprocessing scripts were present in the clone
- switching the clone to remote `dev` still did not expose those scripts

This shows that multiple local preprocessing scripts currently exist only in the local working tree and are not yet part of the remote repository history.

### 0.5. The current molecular `data.pkl` formats are not uniform

The current repository already contains different baseline formats for the three molecular datasets:

- `qm9`
  - sample type: `(DGLGraph, dict)`
  - label payload: 16-property dictionary
  - graph fields: `ndata['pos']`, `ndata['attr']`, `edata['edge_attr']`
- `zinc`
  - sample type: `(DGLGraph, torch.Tensor)`
  - label payload: scalar tensor with shape `(1,)`
  - graph fields: `ndata['feat']`, `edata['feat']`
- `aqsol`
  - sample type: `(DGLGraph, float)`
  - label payload: Python `float`
  - graph fields: `ndata['feat']`, `edata['feat']`

This means that “strictly consistent with existing processed datasets” must be interpreted dataset-by-dataset, not as one single shared molecular format.

### 1. `qm9` and `qm9test` now have a split between raw scaffold and secondary scripts

- `data/qm9/prepare_qm9_raw.py`
- `data/qm9/process_qm9_dataset.py`
- `data/qm9test/create_qm9test_dataset.py`

Current status:

- `data/qm9/prepare_qm9_raw.py` is the raw cold-start scaffold targeting the current baseline layout
- `data/qm9/process_qm9_dataset.py` remains a secondary script built on already processed QM9 data
- `data/qm9test/create_qm9test_dataset.py` now replays directly from canonical `data/qm9`, instead of using a loader path that destroys the original global ordering

Current verified state:

1. `qm9` split files are byte-identical to baseline
2. `qm9` four `smiles_*` side files are byte-identical to baseline
3. `qm9` full sample-by-sample tensor comparison is exact, while `data.pkl` bytes still differ
4. `qm9test` split files, four `smiles_*` side files, and `metadata.json` are byte-identical to baseline
5. `qm9test` `data.pkl` is semantically identical, while pickle bytes still differ

For the deeper raw-data lineage, the backup repository also contains:

- `/home/gzy/py/backup_tokenizerGraph/backup/legacy_scripts/qm9_loader.py`

That legacy implementation appears to contain raw QM9 download and processing logic, but it is not yet normalized into the current repository structure.

### 2. `zinc` and `aqsol` remain experimental and out of this round's formal scope

- The historical molecule preparation flow exists in:
  - `/home/gzy/py/backup_tokenizerGraph/foreign_dataset_files_to_convert/dataset_prepare.py`
  - `/home/gzy/py/backup_tokenizerGraph/foreign_dataset_files_to_convert/molecules.py`

Current status:

1. the repository contains experimental scripts and internal comparison notes for both datasets
2. these scripts are not part of the paper-facing guaranteed cold-start surface for this round
3. the authoritative deliverable scope should therefore exclude `zinc` and `aqsol`

### 2.5. Current runtime transport blockers already identified

- `code2`
  - public source path exists through OGB
  - current runtime encountered `HTTP 502` from the upstream download path
- `qm9`
  - script logic is now validated, but raw-source access still depends on runtime transport/proxy conditions
- `zinc`
  - experimental path only; not part of the current guaranteed scope
- `aqsol`
  - experimental path only; not part of the current guaranteed scope

These are currently treated as environment or upstream transport blockers rather than evidence that the repository-side preprocessing design is wrong.

### 3. Export documentation is noisier than the codebase

Current documentation still references scripts such as:

- `export_qm9.py`
- `export_zinc.py`
- `export_molhiv.py`

These references were found in:

- `docs/guides/dataset_export_guide.md`
- `docs/guides/simple_export_guide.md`
- `export_system/DATASET_EXPORT_GUIDE.md`
- `export_system/SIMPLE_EXPORT_GUIDE.md`

The corresponding scripts were not found in the repository. These references should be treated as documentation drift rather than implemented reproducibility paths.

### 4. Existing processed data remains a comparison baseline

The current processed datasets under `data/<dataset>` are the baseline targets for comparison. Any newly implemented cold-start script must be validated against them for:

- directory structure
- pickle payload structure
- split cardinalities
- auxiliary side files when required

## Immediate Next Actions

1. Continue normalizing the remaining unresolved paper-scope dataset `mnist`.
2. Remove or downgrade non-existent export script references from documentation.
3. Keep `zinc` and `aqsol` documented as experimental drafts, not as current formal deliverables.
