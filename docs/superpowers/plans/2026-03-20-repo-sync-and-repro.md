# Repo Sync And Repro Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 审计并整理当前仓库的数据处理补全工作，完成冷启动复现，修正文档或代码，再同步主线、`dev`、`release` 分支并推送远端。

**Architecture:** 先基于现有分支与测试完成“内容归类”，明确哪些提交属于数据处理补全、哪些属于发布文档或历史清理；再在独立克隆中执行最小复现实验，依据结果修正文档或代码；最后以验证过的提交为基础整理分支关系并推送。所有对外结论必须有新鲜验证证据支撑。

**Tech Stack:** Git, Python 3.10, pytest, virtualenv, 项目现有训练/预处理脚本

---

## Chunk 1: 仓库与分支审计

### Task 1: 归类主线、开发线、发布线差异

**Files:**
- Modify: `docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md`

- [ ] **Step 1: 收集分支差异与提交图**

Run: `git log --oneline --decorate --graph --max-count=25 --all --simplify-by-decoration`
Expected: 能看出 `master`、`dev`、`release`、当前工作分支的分叉点和最近提交。

- [ ] **Step 2: 归纳数据处理相关改动范围**

Run: `git diff --name-only master..dev`
Expected: 能定位新增的数据预处理脚本、测试、README 与配置改动。

- [ ] **Step 3: 写入过程文档**

在过程文档中记录：
- 哪些提交建议进入主线
- 哪些改动只保留在 `release`
- 哪些内容不应纳入本次提交

## Chunk 2: 冷启动复现

### Task 2: 全新克隆并建立独立环境

**Files:**
- Modify: `docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md`

- [ ] **Step 1: 冷启动克隆**

Run: `git clone <repo> /tmp/tokenizerGraph-coldstart-20260320`
Expected: 获得干净仓库，并记录默认检出分支。

- [ ] **Step 2: 采集资源与环境信息**

Run: `python -V`, `nproc`, `free -h`, `df -h /tmp`, `nvidia-smi`
Expected: 记录 Python、CPU、内存、磁盘和 GPU 可见性。

- [ ] **Step 3: 创建独立虚拟环境并安装**

Run: `python -m venv .venv` 与 `./.venv/bin/pip install -e .`
Expected: 如果失败，记录失败点和缺失依赖；如果成功，进入最小验证。

### Task 3: 执行最小复现验证

**Files:**
- Modify: `docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md`
- Test: `tests/test_reproducibility_documentation.py`
- Test: `tests/test_data_preprocess_script_inventory.py`
- Test: `tests/test_qm9_raw_script_scaffold.py`
- Test: `tests/test_qm9_lineage.py`
- Test: `tests/test_repro_compare.py`

- [ ] **Step 1: 运行最小文档与脚本验证集**

Run: `pytest tests/test_reproducibility_documentation.py tests/test_data_preprocess_script_inventory.py tests/test_qm9_raw_script_scaffold.py tests/test_qm9_lineage.py tests/test_repro_compare.py -v`
Expected: 验证冷启动文档、脚本清单与 `qm9` 数据链路约束。

- [ ] **Step 2: 运行文档声明的最小 smoke 命令**

Run: `prepare_data_new.py -> run_pretrain.py -> run_finetune.py` 的 `qm9test` 最小链路
Expected: 如成功则记录关键输出路径；如失败则记录首个真实阻塞点。

## Chunk 3: 修复与收口

### Task 4: 修正文档或代码使复现可用

**Files:**
- Modify: `README.md`
- Modify: `README_zh.md`
- Modify: `scripts/dataset_conversion/README.md`
- Modify: `docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md`
- Modify: 复现失败指向的最小代码集（如必要）

- [ ] **Step 1: 先写失败现象与根因**

在过程文档记录失败命令、异常位置、受影响分支与最小修复目标。

- [ ] **Step 2: 做最小修复**

仅修改能真实解除冷启动阻塞的代码或文档，避免无关整理。

- [ ] **Step 3: 重新运行对应验证**

Run: 失败过的同一条命令或最小测试集
Expected: 验证阻塞已解除，或明确剩余外部依赖不可控。

## Chunk 4: 分支整理、提交与推送

### Task 5: 整理主线、开发线、发布线

**Files:**
- Modify: `docs/reproducibility/2026-03-20-repo-sync-and-repro-log.md`

- [ ] **Step 1: 形成最终提交**

Run: `git status --short`
Expected: 只包含本次确认需要提交的代码和文档。

- [ ] **Step 2: 先验证再提交**

Run: 与修改内容对应的 pytest/脚本命令
Expected: 有新鲜通过证据后再提交。

- [ ] **Step 3: 提交并同步分支**

Run: 非交互式 `git add` / `git commit` / `git checkout` / `git merge`
Expected: `master`、`dev`、`release` 达成目标状态，且无意外文件混入。

- [ ] **Step 4: 推送远端**

Run: `git push origin <branch>`
Expected: 远端分支更新成功。
