# 未使用代码扫描报告

> **生成时间**: 2025-10-30  
> **原则**: 暂时保留可能未使用的代码，仅在文档中标注状态

## 一、数据集加载器状态

### ✅ 已注册且使用的加载器

| 数据集 | 加载器文件 | 状态 | 备注 |
|--------|-----------|------|------|
| `qm9` | `qm9_loader.py` | ✅ 常用 | 主要数据集 |
| `qm9test` | `qm9test_loader.py` | ✅ 常用 | 测试数据集 |
| `zinc` | `zinc_loader.py` | ✅ 已注册 | 分子数据集 |
| `aqsol` | `aqsol_loader.py` | ✅ 已注册 | 溶解度数据集 |
| `mnist` | `mnist_loader.py` | ✅ 已注册 | 图像数据集 |
| `mutagenicity` | `mutagenicity_loader.py` | ✅ 已注册 | 分类数据集 |
| `proteins` | `proteins_loader.py` | ✅ 已注册 | 蛋白质数据集 |
| `peptides_func` | `peptides_func_loader.py` | ✅ 已注册 | 肽功能预测 |
| `peptides_struct` | `peptides_struct_loader.py` | ✅ 已注册 | 肽结构预测 |
| `molhiv` | `molhiv_loader.py` | ✅ 已注册 | MoleculeNet HIV |

### ⚠️ 已注册但较少使用的加载器

| 数据集 | 加载器文件 | 状态 | 备注 |
|--------|-----------|------|------|
| `mnist_raw` | `mnist_raw_loader.py` | ⚠️ 已注册 | 可能需要保留 |
| `colors3` | `colors3_loader.py` | ⚠️ 已注册 | 较少使用 |
| `synthetic` | `synthetic_loader.py` | ⚠️ 已注册 | 合成数据 |
| `coildel` | `coildel_loader.py` | ⚠️ 已注册 | 较少使用 |
| `dblp` | `dblp_loader.py` | ⚠️ 已注册 | 网络数据集 |
| `dd` | `dd_loader.py` | ⚠️ 已注册 | 较少使用 |
| `twitter` | `twitter_loader.py` | ⚠️ 已注册 | 网络数据集 |
| `code2` | `code2_loader.py` | ⚠️ 已注册 | 代码数据集 |

**建议**: 保留所有已注册的加载器，它们可能在未来实验中使用。

## 二、序列化器状态

### ✅ 已注册的序列化器

| 方法名 | 实现文件 | 状态 | 备注 |
|--------|---------|------|------|
| `feuler` | `freq_eulerian_serializer.py` | ✅ 主要方法 | 推荐使用 |
| `eulerian` | `eulerian_serializer.py` | ✅ 常用 | 标准欧拉 |
| `cpp` | `chinese_postman_serializer.py` | ✅ 已注册 | 中国邮路 |
| `fcpp` | `freq_chinese_postman_serializer.py` | ✅ 已注册 | 频率邮路 |
| `dfs` | `dfs_serializer.py` | ✅ 已注册 | 深度优先 |
| `bfs` | `bfs_serializer.py` | ✅ 已注册 | 广度优先 |
| `topo` | `topo_serializer.py` | ✅ 已注册 | 拓扑排序 |
| `smiles` | `smiles_serializer.py` | ✅ 已注册 | SMILES系列 |
| `smiles_1-4` | `smiles_serializer.py` | ✅ 已注册 | SMILES变体 |
| `image_row_major` | `image_row_major_serializer.py` | ⚠️ 已注册 | 图像专用 |
| `image_serpentine` | `image_serpentine_serializer.py` | ⚠️ 已注册 | 图像专用 |
| `image_diag_zigzag` | `image_diag_zigzag_serializer.py` | ⚠️ 已注册 | 图像专用 |

**建议**: 保留所有序列化器，它们针对不同类型的数据有各自的用途。

## 三、根目录脚本文件状态

### ✅ 核心脚本（必须保留）

| 文件 | 状态 | 说明 |
|------|------|------|
| `run_pretrain.py` | ✅ 核心 | 预训练入口 |
| `run_finetune.py` | ✅ 核心 | 微调入口 |
| `prepare_data_new.py` | ✅ 核心 | 数据预处理 |
| `config.py` | ✅ 核心 | 配置管理 |

### ⚠️ 测试和工具脚本

| 文件 | 状态 | 说明 | 建议 |
|------|------|------|------|
| `test_gte_integration.py` | ⚠️ 保留 | GTE集成测试 | 重要测试 |
| `test_serialization_methods.py` | ⚠️ 保留 | 序列化方法测试 | 重要测试 |
| `test.py` | ❓ 需检查 | 通用测试 | 查看内容 |
| `process.py` | ❓ 需检查 | 处理脚本 | 查看内容 |
| `aggregate_results.py` | ⚠️ 工具 | 结果聚合 | 可能有用 |
| `analyze_export_datasets_stats.py` | ⚠️ 工具 | 数据分析 | 可能有用 |
| `analyze_lrgb_datasets.py` | ⚠️ 工具 | 数据集分析 | 可能有用 |
| `benchmark_serialization_speed.py` | ⚠️ 工具 | 性能测试 | 可能有用 |

### ❌ 可能废弃的脚本

| 文件 | 状态 | 说明 | 建议 |
|------|------|------|------|
| `batch_finetune_simple.py` | ❓ 需检查 | 批量微调 | 查看是否仍在使用 |
| `batch_pretrain_simple.py` | ❓ 需检查 | 批量预训练 | 查看是否仍在使用 |
| `slurm_submit_simple.py` | ⚠️ 工具 | SLURM提交脚本 | 可能有用 |
| `batch_submit_clearml.py` | ⚠️ 工具 | ClearML批量提交 | 可能有用 |

### 📝 Shell脚本

| 文件 | 状态 | 说明 |
|------|------|------|
| `*.sh` | ⚠️ 保留 | 实验脚本，可能包含重要配置 |

**建议**: 保留所有shell脚本，它们可能包含实验的具体配置。

## 四、文档文件状态

### ✅ 核心文档（已验证）

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/data/README.md` | ✅ 已更新 | 数据层文档 |
| `src/algorithms/serializer/README.md` | ✅ 已更新 | 序列化文档 |
| `src/algorithms/compression/README.md` | ✅ 已验证 | BPE文档 |
| `src/models/bert/README.md` | ⚠️ 需检查 | 可能过时 |
| `TokenizerGraph_Detailed_Documentation.md` | ✅ 已更新 | 主技术文档 |

### ⚠️ 专项文档

| 文件 | 状态 | 说明 | 建议 |
|------|------|------|------|
| `CONFIG_GUIDE.md` | ✅ 保留 | 配置指南 | 重要 |
| `EXPERIMENT_GUIDE.md` | ✅ 保留 | 实验指南 | 重要 |
| `CURRENT_PROJECT_STATUS.md` | ⚠️ 需更新 | 项目状态 | 可能需要更新 |
| `CODING_STANDARDS.md` | ✅ 保留 | 编码规范 | 重要 |

### 📦 设计和规划文档（归档候选）

| 文件 | 状态 | 说明 | 建议 |
|------|------|------|------|
| `WORKFLOW_COMPARISON.md` | 📦 归档 | 工作流对比 | 移至`docs/archive/` |
| `UNIFIED_MODEL_PROGRESS.md` | 📦 归档 | 模型进度 | 移至`docs/archive/` |
| `UNIVERSAL_MODEL_REFACTOR_DESIGN.md` | 📦 归档 | 重构设计 | 移至`docs/archive/` |
| `PIPELINE_REFACTOR_PLAN.md` | 📦 归档 | 流水线计划 | 移至`docs/archive/` |
| `REFACTOR_ISSUES_AND_SOLUTIONS.md` | 📦 归档 | 重构问题 | 移至`docs/archive/` |
| `NEXT_STEPS.md` | 📦 归档 | 下一步计划 | 移至`docs/archive/` |
| `BATCH_SCRIPT_UPDATES.md` | 📦 归档 | 批量脚本更新 | 移至`docs/archive/` |

### 📝 导出和数据集文档

| 文件 | 状态 | 说明 |
|------|------|------|
| `DATASET_EXPORT_GUIDE.md` | ✅ 保留 | 数据集导出指南 |
| `SIMPLE_EXPORT_GUIDE.md` | ✅ 保留 | 简单导出指南 |
| `DGL_graph_pred_discrete_datasets.md` | ⚠️ 需检查 | DGL数据集文档 |

## 五、模型模块状态

### ✅ 核心模型

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/models/bert/` | ✅ 使用中 | BERT模型 |
| `src/models/gte/` | ✅ 使用中 | GTE模型 |
| `src/models/unified_encoder.py` | ✅ 使用中 | 统一编码器 |
| `src/models/unified_task_head.py` | ✅ 使用中 | 统一任务头 |
| `src/models/universal_model.py` | ✅ 使用中 | 通用模型 |

### ⚠️ 其他模块

| 模块 | 状态 | 说明 | 建议 |
|------|------|------|------|
| `src/models/gnn/` | ⚠️ 需检查 | GNN模型 | 查看是否仍在使用 |
| `src/models/aggregators/` | ⚠️ 需检查 | 聚合器 | 查看是否仍在使用 |

## 六、辅助工具模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/utils/` | ✅ 使用中 | 工具模块 |
| `src/utils/check.py` | ✅ 使用中 | 检查工具 |
| `src/utils/config_override.py` | ✅ 使用中 | 配置覆盖 |
| `src/utils/logger.py` | ✅ 使用中 | 日志工具 |
| `src/utils/metrics.py` | ✅ 使用中 | 评估指标 |

## 七、训练模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `src/training/pretrain_pipeline.py` | ✅ 使用中 | 预训练流水线 |
| `src/training/finetune_pipeline.py` | ✅ 使用中 | 微调流水线 |
| `src/training/model_builder.py` | ✅ 使用中 | 模型构建 |
| `src/training/task_handler.py` | ✅ 使用中 | 任务处理 |
| `src/training/tasks.py` | ✅ 使用中 | 任务定义 |

## 八、建议的行动

### 立即行动

1. ✅ **已完成**: 更新核心README文档
2. ✅ **已完成**: 验证主技术文档
3. ⏳ **进行中**: 扫描未使用代码

### 后续行动

1. **检查 `src/models/bert/README.md`**: 确认是否仍适用，如不适用则更新或删除
2. **更新 `CURRENT_PROJECT_STATUS.md`**: 反映当前实际状态
3. **归档设计文档**: 将历史设计文档移至`docs/archive/`
4. **检查测试脚本**: 确认`test.py`和`process.py`的用途
5. **代码注释**: 为核心模块添加/更新docstring

### 保留原则

- ✅ **已注册的加载器/序列化器**: 全部保留（可能在未来实验中使用）
- ✅ **核心脚本**: 全部保留
- ⚠️ **工具脚本**: 暂时保留，标注状态
- 📦 **设计文档**: 移至归档目录

---

**更新频率**: 建议每季度更新一次  
**维护者**: 发现实际未使用的代码时，在此文档中标注状态

