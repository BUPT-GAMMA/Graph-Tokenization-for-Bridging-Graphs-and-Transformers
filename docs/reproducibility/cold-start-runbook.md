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
