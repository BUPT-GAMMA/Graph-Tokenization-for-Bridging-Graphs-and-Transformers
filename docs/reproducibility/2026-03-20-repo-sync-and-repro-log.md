# 仓库整理、分支同步与复现过程记录

## 原始目标

- 梳理当前仓库状态，确认数据处理部分已补全的工作内容。
- 将需要提交的代码整理为规范的 Git 提交，并合并到主分支。
- 整理 `release` 和 `dev` 两个分支，确保二者都基于最新代码、包含应保留的修改，并符合各自定位。
- 从零克隆整个仓库，在新环境中尝试复现，记录问题并修复文档或代码，使复现流程真实可用。
- 最终推送到 GitHub。

## 当前理解

- 远端不存在 `main`；冷启动克隆显示当前远端默认分支是 `release`，而当前工作仓库保存的 `origin/master` 默认分支信息已经过期。
- 本地当前工作分支为 `repro-audit-local`。
- 本地存在未跟踪内容：`a.txt`、`data/gnn_use/`，暂不纳入本次整理范围。
- `dev` 分支已有一轮数据准备审计与冷启动文档补充；`release` 分支承载面向发布展示的 README/文档整理。

## 执行计划

1. 审计 `master`、`dev`、`release`、当前工作分支之间的差异，确定哪些提交应进入主线。
2. 在隔离目录从零克隆仓库，建立干净环境，验证安装与最小数据处理链路。
3. 根据复现结果修订代码或文档，补足缺失依赖说明、命令、故障排查与约束。
4. 将确认后的改动提交到合适分支，整理 `master`/`dev`/`release` 的相互关系。
5. 推送到远端，并产出最终总结。

## 当前状态

- 已读取全局/项目协作规则与本次任务相关流程技能。
- 已确认远端为 `origin git@github.com:BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers.git`。
- 已初步拿到 `master..dev`、`master..release` 的统计差异，待进一步归类。
- 已在 `/tmp/tokenizerGraph-coldstart-20260320` 完成一次全新克隆，用于冷启动复现。

## 已发现问题

- 远端主线名与用户描述不一致：远端没有 `main`，而且默认检出分支是 `release`，不是当前工作仓库里陈旧记录指向的 `master`。
- `dev` 与 `release` 相对 `master` 的改动规模都较大，且性质不同，需要先做内容归类，再决定合并策略。
- 仓库根目录包含大量实验产物目录与历史文件，复现时需要明确哪些是必需输入，哪些只是本地缓存/产物。
- 冷启动克隆的默认检出分支实际是 `release`，且该克隆内 `origin/HEAD -> origin/release`；当前工作仓库里的 `origin/master` 记录是陈旧状态，不能继续拿它当远端默认分支依据。
- 新建虚拟环境后执行 `pip install -e .` 会在元数据阶段失败：`setup.py` 直接导入 `pybind11`，但仓库此前没有声明构建依赖，导致干净环境无法完成可编辑安装。

## 环境快照

- 冷启动环境 Python：`3.10.9`
- CPU 逻辑核心数：`128`
- 内存：总计约 `251 GiB`，可用约 `205 GiB`
- `/tmp` 所在磁盘可用空间：约 `276 GiB`
- `nvidia-smi` 在当前沙箱中返回 `Failed to initialize NVML: Unknown Error`，说明此会话下 GPU 可见性需要单独验证，不能默认认为冷启动环境具备可用 CUDA

## 已实施修复

- 新增 `pyproject.toml`，声明 `setuptools`、`wheel`、`pybind11` 为构建依赖，使 `pip install -e .` 不再依赖手工预装 `pybind11`。
- 在 `README.md` 与 `README_zh.md` 的安装章节补充说明：
  - 联网环境下 `pip install -e .` 会自动补齐构建依赖
  - 离线环境需要先预装 `pybind11`
  - 运行实验仍需环境中具备 `torch`、`dgl`、`rdkit`、`transformers` 等运行时依赖
- 新增回归测试，约束仓库必须显式声明 `pybind11` 构建依赖。
- 新增 `docs/reproducibility/environment-setup.md`，明确区分：
  - 构建依赖由 `pip install -e .` 自动处理
  - 运行时依赖需要用户先准备
  - 当前训练链路仍受 CUDA 可见性约束
- 在 `docs/reproducibility/paper-dataset-cold-start-guide.md` 中补充环境准备入口，并把 README 中的复现资源列表链接到该文档。

## 新鲜验证结果

- 最小回归测试集已通过：
  - `tests/test_reproducibility_documentation.py`
  - `tests/test_data_preprocess_script_inventory.py`
  - `tests/test_qm9_raw_script_scaffold.py`
  - `tests/test_qm9_lineage.py`
  - `tests/test_repro_compare.py`
- 在全新虚拟环境 `/tmp/tokenizerGraph-installcheck-20260320/.venv` 中，以下命令成功完成：
  - `pip install -e /home/gzy/py/tokenizerGraph`
  - 其中 build dependencies、editable metadata、C++ 扩展构建全部成功
- 基于提交 `45c9b748e0e1672e7b8d17fd4ea07ad45352ba00` 的干净本地克隆 `/tmp/tokenizerGraph-postcommit-20260320` 中：
  - `pip install -e /tmp/tokenizerGraph-postcommit-20260320` 再次成功
  - `ProjectConfig().cache_dir` 解析为仓库内本地路径 `/tmp/tokenizerGraph-postcommit-20260320/cache`
  - 证明此前 `/local/gzy/tokg` 权限问题来自当前工作树的本地 `cache` 符号链接，而非仓库默认配置

## 复现边界与新增发现

- 新虚拟环境中直接导入项目包仍会因为缺少 `torch` 等运行时依赖而失败；这不是新的构建失败，而是仓库当前并非“一条 pip 命令自动装全运行时”的完整 Python 包。
- `prepare_data_new.py --datasets qm9test --methods feuler --bpe_merges 50` 在当前工作树失败，但根因不是仓库默认配置：
  - 当前工作树存在一个未被 Git 跟踪的本地符号链接 `cache -> /local/gzy/tokg/data`
  - `config.py` 将相对路径 `cache/` 解析为该符号链接目标，导致在本地环境中触发 `PermissionError`
  - 全新克隆目录中并不存在该 `cache` 符号链接，因此这一错误应记录为“本地环境污染”，而不是冷启动仓库级缺陷
- 当前会话下：
  - `nvidia-smi` 返回 `Failed to initialize NVML: Unknown Error`
  - `torch.cuda.is_available()` 返回 `False`
  - 因此本轮不能对 `run_finetune.py` 的 CUDA 路径作通过声明
- 在干净克隆中执行 `python prepare_data_new.py --datasets qm9test --methods feuler --bpe_merges 50` 后，首次真实失败点变为：
  - `data/qm9test/train_index.json` 等索引文件不存在
  - 说明 `qm9test` 不是 clean clone 自带的 checked-in 示例数据，而是需要先由 `qm9` 派生得到的 smoke-test 数据集
  - 已据此修正 README 的对外口径，避免继续把 `qm9test` 写成仓库内现成样例

## 下一步

- 继续分析关键分支的提交与核心文件差异，聚焦数据处理链路相关改动。
- 形成干净提交，并在新的 Git 克隆中基于当前提交重新验证安装/分支行为。
- 随后整理 `master`、`dev`、`release` 的同步策略并推送远端。
