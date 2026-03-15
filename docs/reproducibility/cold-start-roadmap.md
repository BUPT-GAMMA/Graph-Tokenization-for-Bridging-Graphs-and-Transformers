# Cold-Start Roadmap

## Final Goal

对项目实际用到的每一个数据集，实现如下闭环：

1. 从公开原始数据冷启动。
2. 生成当前 loader 所需的 `data/<dataset>` 目录结构。
3. 生成结果与当前项目既有处理后结果严格一致，至少满足：
   - 文件集合一致
   - split 长度一致
   - `data.pkl` 样本结构一致
   - 必要时字节级或哈希级一致

## Phase Breakdown

### Phase 1: Repository readiness

目标：

- 让冷启动脚本本身成为仓库的一部分，而不是本地隐形文件。
- 修正文档与实际脚本不一致的问题。

状态：

- 已完成

已完成项：

- 修正 `.gitignore`，允许跟踪 `data/<dataset>/` 下的源码脚本
- 将现有预处理脚本纳入本地仓库分支
- 删除文档中把不存在导出脚本当成已实现入口的说法

### Phase 2: Audit and baseline locking

目标：

- 锁定当前处理后数据的 baseline 结构
- 锁定每个数据集的冷启动状态

状态：

- 已完成第一轮

已完成项：

- `docs/reproducibility/dataset-cold-start-audit.md`
- `tests/test_data_preprocess_script_inventory.py`
- `tests/test_molecular_dataset_current_format.py`

### Phase 3: Clone-based reproduction

目标：

- 在独立克隆目录里验证冷启动，而不是只在当前工作区运行

状态：

- 已部分完成

已完成项：

- `mnist_raw` 冷启动成功
- 与当前 baseline 字节级一致

未完成项：

- 其余数据集尚未逐个执行 clone-based reproduction

### Phase 4: Dataset-specific closure

目标：

- 每个数据集都形成：
  - 正式冷启动脚本
  - 对比验证方法
  - 文档入口

状态：

- 进行中

## Current Dataset Status

### Completed end-to-end at cold-start level

- `mnist_raw`

### Clone-verified with semantic equivalence but not byte-identical `data.pkl`

- `molhiv`
- `proteins`

### Cold-start runs but currently mismatches baseline

- `colors3`

### Script exists and can likely be reproduced next

- `molhiv`
- `code2`
- `proteins`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dblp`
- `dd`
- `twitter`
- `peptides_func`
- `peptides_struct`

### Needs normalization before true public-raw cold-start

- `qm9`
- `qm9test`
- `zinc`
- `aqsol`
- `mnist`

## Required Remaining Work

### Group A: Existing cold-start scripts, not yet clone-verified

For each dataset in this group:

1. clone-based run
2. compare against existing baseline
3. write result back to runbook/audit

Datasets:

- `molhiv`
- `code2`
- `colors3`
- `proteins`
- `synthetic`
- `mutagenicity`
- `coildel`
- `dblp`
- `dd`
- `twitter`
- `peptides_func`
- `peptides_struct`

### Group B: Molecular pipeline recovery

For these datasets, the current scripts are not yet true public-raw cold-start entry points:

- `qm9`
- `qm9test`
- `zinc`
- `aqsol`

Needed work:

1. recover the raw-data lineage from backup scripts
2. normalize into current repository
3. export current-format artifacts
4. compare against existing baseline

### Group C: MNIST superpixel normalization

- `mnist`

Needed work:

1. isolate the `final_slic` dependency path
2. make the preprocessing entry point self-contained
3. run clone-based comparison

## Execution Order

Recommended order from now on:

1. `molhiv`
2. one TU dataset (`colors3` or `proteins`)
3. one LRGB dataset (`peptides_func`)
4. `code2`
5. `qm9`
6. `qm9test`
7. `zinc`
8. `aqsol`
9. `mnist`

## Evidence Files

- `docs/reproducibility/dataset-cold-start-audit.md`
- `docs/reproducibility/cold-start-runbook.md`
- `tests/test_reproducibility_documentation.py`
- `tests/test_data_preprocess_script_inventory.py`
- `tests/test_molecular_dataset_current_format.py`
