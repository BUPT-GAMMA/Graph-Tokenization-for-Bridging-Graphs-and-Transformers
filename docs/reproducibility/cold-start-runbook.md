# Cold-Start Runbook

## Objective

在独立克隆目录中，从公开原始数据开始执行一次冷启动复现，并将生成结果与当前仓库中的既有处理后数据进行结构对比。

## Formal Scope For This Round

本 runbook 仅将论文主结果实际涉及的数据集视为“正式保证范围”。

当前不纳入正式保证范围的数据集：

- `zinc`
- `aqsol`
- `mnist`

说明：

- `zinc` 和 `aqsol` 在仓库中保留实验性脚本与内部验证记录
- 但本轮对外口径不再将其计为“正式已实现的冷启动流程”
- `mnist` 仍依赖未标准化的 `final_slic` 路径，因此继续保留为未完成项

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

## Successful Clone-Based Reproduction in Local Snapshot

To continue verification without waiting for the remote repository to include those scripts, a local snapshot branch was created and cloned:

- source branch: `repro-audit-local`
- source commit used for clone: `70b82a2`
- clone path: `/tmp/tokenizerGraph-cold-start-local`

### Dataset

- `mnist_raw`

### Command

```bash
python data/mnist_raw/prepare.py
```

### Outcome

- public download succeeded through the current proxy environment
- `data/mnist_raw/data.pkl` was generated with `70000` samples
- split files were generated:
  - `train_index.json` -> `56000`
  - `val_index.json` -> `7000`
  - `test_index.json` -> `7000`
- the script's built-in validation completed successfully

### Comparison Against Existing Baseline

The newly generated files were compared against the current repository baseline at `/home/gzy/py/tokenizerGraph/data/mnist_raw`.

SHA-256 comparison:

- `data.pkl`
  - baseline: `f972168e3a211fd665a307ad37dba63f45a10dd3f4150523b787d8f316d647c1`
  - cloned run: `f972168e3a211fd665a307ad37dba63f45a10dd3f4150523b787d8f316d647c1`
- `train_index.json`
  - baseline: `c066c82580c2cf26fed2730306f5e50f0f1265e4f31806f9878d94ef31f2f4d2`
  - cloned run: `c066c82580c2cf26fed2730306f5e50f0f1265e4f31806f9878d94ef31f2f4d2`
- `val_index.json`
  - baseline: `f94831ef28a8c6576976962827f4c6aaab2f67b5c508c4dcd12ca4162c7c7204`
  - cloned run: `f94831ef28a8c6576976962827f4c6aaab2f67b5c508c4dcd12ca4162c7c7204`
- `test_index.json`
  - baseline: `e2e0956c699938b02f15276a22960788ef94d32f0fcd5a7289473274262b1745`
  - cloned run: `e2e0956c699938b02f15276a22960788ef94d32f0fcd5a7289473274262b1745`

Additional checks:

- `data.pkl` sample count matched exactly
- first sample label and image array matched exactly
- last sample label and image array matched exactly
- `UnifiedDataInterface` in the cloned repository loaded the generated dataset successfully

### Conclusion

`mnist_raw` is now verified as a clone-based cold-start reproducible dataset, and its generated artifacts are byte-identical to the current baseline.

## Additional Clone-Based Results

### `molhiv`

Command:

```bash
python data/molhiv/preprocess_molhiv.py
```

Observed result:

- public OGB download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` matched the current baseline byte-for-byte
- `data.pkl` did not match at raw file hash level
- `data.pkl` did match at semantic digest level
- sampled graph/label checks at index `0`, `20563`, `41126` were all exactly equal in:
  - labels
  - node count
  - edge count
  - `ndata` keys, shapes, dtypes, values
  - `edata` keys, shapes, dtypes, values

Interpretation:

- current cold-start script reproduces the same dataset semantics
- the remaining difference is in pickle-level binary representation, not the dataset content checked so far

### `proteins`

Command:

```bash
python data/proteins/preprocess_proteins.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` matched the current baseline byte-for-byte
- `data.pkl` did not match at raw file hash level
- `data.pkl` did match at semantic digest level

Interpretation:

- same situation as `molhiv`
- split policy matches baseline exactly
- pickle bytes still differ

### `colors3`

Command:

```bash
python data/colors3/preprocess_colors3.py
```

Observed result after split-policy fix:

- TU download succeeded
- all three split files now match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash still differs

Interpretation:

- `colors3` has been upgraded from “split mismatch” to the same status as `molhiv` and `proteins`
- remaining difference is in pickle-level binary representation only

### `peptides_func`

Command:

```bash
python data/peptides_func/prepare_lrgb_data.py
```

Observed result:

- public LRGB download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl.gz` semantic digest matches the current baseline
- `data.pkl.gz` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in compressed pickle bytes

### `peptides_struct`

Command:

```bash
python data/peptides_func/prepare_lrgb_data.py
```

Observed result:

- public LRGB download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl.gz` semantic digest matches the current baseline
- `data.pkl.gz` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in compressed pickle bytes

### `synthetic`

Command:

```bash
python data/synthetic/preprocess_synthetic.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

### `mutagenicity`

Command:

```bash
python data/mutagenicity/preprocess_mutagenicity.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

### `coildel`

Command:

```bash
python data/coildel/preprocess_coil_del.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

### `dd`

Command:

```bash
python data/dd/preprocess_dd.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

### `dblp`

Command:

```bash
python data/dblp/preprocess_dblp_v1.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

### `twitter`

Command:

```bash
python data/twitter/preprocess_twitter_real_graph_partial.py
```

Observed result:

- TU download succeeded
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- `data.pkl` semantic digest matches the current baseline
- `data.pkl` raw file hash differs

Interpretation:

- current cold-start script reproduces the same dataset semantics and split policy
- remaining difference is in pickle-level binary representation only

## Current External Source Blockers

### `code2`

Attempted command:

```bash
python data/code2/preprocess_code2.py
```

Observed blocker:

- OGB upstream download returned `HTTP 502` during `DglGraphPropPredDataset(name="ogbg-code2")`

Interpretation:

- this is currently an external-source availability issue
- it is not yet evidence of a repository-side preprocessing bug

### `qm9`

Command used for the latest verified replay:

```bash
python data/qm9/prepare_qm9_raw.py \
  --split-source-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --reference-data-pkl /home/gzy/py/tokenizerGraph/data/qm9/data.pkl \
  --reference-smiles-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --output-dir /tmp/qm9-raw-run
```

Raw-source probes used during script development:

1. DGL source:

```text
https://data.dgl.ai/dataset/qm9_edge.npz
```

2. DeepChem CSV source:

```text
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv
```

Current progress:

- the raw script now uses `dgl.data.QM9EdgeDataset`
- `train_index.json`, `val_index.json`, `test_index.json` can be reproduced byte-for-byte
- the four `smiles_*` side files are now byte-identical when replaying the current baseline side files
- a full sample-by-sample tensor scan now reports `all_equal`
- `data.pkl` still differs at raw pickle-byte level, but now reports `semantic_match=true`

Interpretation:

- the remaining `data.pkl` difference is now at pickle-byte level only
- the earlier duplicate-signature alignment bug around sample `4699` has been resolved by preferring exact graph-tensor signatures before WL fallback

### `qm9test`

Command used for the latest verified replay:

```bash
python data/qm9test/create_qm9test_dataset.py \
  --original-indices-path /home/gzy/py/tokenizerGraph/data/qm9test/metadata.json \
  --source-dir /home/gzy/py/tokenizerGraph/data/qm9 \
  --output-dir /tmp/qm9test-replay
```

Observed result:

- `create_qm9test_dataset.py` now replays directly from `data/qm9`, instead of using a loader path that destroys the original global ordering
- `train_index.json`, `val_index.json`, `test_index.json` are byte-identical to the current baseline
- all four `smiles_*` side files are byte-identical to the current baseline
- `metadata.json` is byte-identical to the current baseline when replaying from the current `metadata.json`
- `data.pkl` is not byte-identical, but reports `semantic_match=true`

Interpretation:

- `qm9test` is now a stable secondary replay step from the canonical `qm9` baseline
- the only remaining artifact-level difference is the same pickle-byte non-determinism already seen in other datasets

### `zinc`

Command:

```bash
export http_proxy=http://local.nginx.show:7890
export https_proxy=http://local.nginx.show:7890
python data/zinc/prepare_zinc_raw.py \
  --use-env-proxy \
  --split-source-dir /home/gzy/py/tokenizerGraph/data/zinc \
  --reference-data-pkl /home/gzy/py/tokenizerGraph/data/zinc/data.pkl \
  --output-dir /tmp/zinc-pkl-run2
```

Observed result:

- direct outbound access to Dropbox and Hugging Face failed in the current runtime
- the local proxy `http://local.nginx.show:7890` was verified to work for both Dropbox and Hugging Face
- the script now uses public `ZINC.pkl` from `https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl`
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- the four `smiles_*` side files match the current baseline byte-for-byte
- `data.pkl` is not byte-identical, but now reports `semantic_match=true`

Interpretation:

- the previous `molecules.zip` path was not the right baseline lineage for the current project
- the public `ZINC.pkl` package plus explicit-h graph normalization reproduces the current baseline semantics
- the remaining difference is pickle-byte level only

### `aqsol`

Command:

```bash
export http_proxy=http://local.nginx.show:7890
export https_proxy=http://local.nginx.show:7890
python data/aqsol/prepare_aqsol_raw.py \
  --use-env-proxy \
  --output-dir /tmp/aqsol-proxy-run2
```

Observed result:

- the same local proxy `http://local.nginx.show:7890` was required for public-source access
- the raw zip dictionaries had to be honored as categorical atom/bond vocabularies, not atomic numbers directly
- RDKit reconstruction required the same relaxed sanitize strategy as the upstream `asqol_conversion_pipeline_v5.py`
- invalid raw samples had to be skipped exactly as in the upstream conversion pipeline
- `train_index.json`, `val_index.json`, `test_index.json` match the current baseline byte-for-byte
- the four `smiles_*` side files match the current baseline byte-for-byte
- `data.pkl` is not byte-identical, but now reports `semantic_match=true`

Interpretation:

- the previous blocker was not source reachability alone; the real issue was reconstruction logic
- current script logic now reproduces the current baseline semantics from public raw inputs
- the remaining difference is pickle-byte level only

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
