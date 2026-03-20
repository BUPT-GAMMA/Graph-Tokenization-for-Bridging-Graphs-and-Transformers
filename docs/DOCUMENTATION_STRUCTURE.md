# 文档结构说明

> **最后更新**: 2025-10-30  
> **命名规范**: 已统一

## 文档目录结构

```
tokenizerGraph/
├── README.md                                    # 项目主README
├── TokenizerGraph_Detailed_Documentation.md    # 主技术文档
│
├── docs/                                        # 文档中心
│   ├── README.md                               # 文档索引
│   ├── unused_code_scan.md                     # 代码扫描报告
│   │
│   ├── guides/                                 # 指南文档
│   │   ├── config_guide.md                    # 配置指南
│   │   ├── experiment_guide.md               # 实验指南
│   │   ├── coding_standards.md               # 编码规范
│   │   ├── dataset_export_guide.md           # 数据集导出
│   │   ├── simple_export_guide.md            # 简单导出
│   │   └── dgl_graph_pred_discrete_datasets.md
│
├── hyperopt/                                   # 超参数搜索工作流与主文档
│   ├── README.md                              # 主说明文档
│   ├── scripts/                               # Optuna 搜索脚本
│   ├── journal/                               # 搜索日志（不提交）
│   └── results/                               # 搜索结果（不提交）
│   │
│   ├── archive/                                # 归档文档（历史）
│   │   ├── README.md                          # 归档索引
│   │   ├── workflow_comparison.md
│   │   ├── unified_model_progress.md
│   │   ├── universal_model_refactor_design.md
│   │   ├── pipeline_refactor_plan.md
│   │   ├── refactor_issues_and_solutions.md
│   │   ├── next_steps.md
│   │   ├── batch_script_updates.md
│   │   ├── current_project_status.md
│   │   ├── model_architecture.md
│   │   └── TokenizerGraph_HighLevel_Summary.md
│   │
│   ├── modules/                                # 模块专项文档
│   │   ├── data/
│   │   │   └── data_layer_new.md              # 数据层详细设计
│   │   └── gte/
│   │       └── gte_integration_guide.md        # GTE集成指南
│   │
│   ├── bpe/                                    # BPE相关文档（保留原有结构）
│   │   ├── README.md
│   │   ├── BPE_USAGE_GUIDE.md
│   │   └── ...
│   │
│   └── PIPELINE_REFACTORING_SUMMARY.md        # 流水线重构总结
│
└── src/                                        # 源代码（模块README在各自目录）
    ├── data/
    │   └── README.md                           # 数据层使用文档
    ├── algorithms/
    │   ├── serializer/
    │   │   └── README.md                      # 序列化文档
    │   └── compression/
    │       └── README.md                       # BPE文档
    └── models/
        ├── bert/
        │   └── README.md                      # BERT文档（历史）
        └── gte/
            └── *.md                            # GTE相关文档
```

## 命名规范

### 根目录文档
- **主README**: `README.md` (小写)
- **主技术文档**: `PascalCase.md` 或 `Title_Case.md`
- **专项报告**: `snake_case.md`

### docs/目录文档
- **指南文档**: `snake_case.md` (docs/guides/)
- **归档文档**: 保持原名 (docs/archive/)
- **模块文档**: `snake_case.md` (docs/modules/)

### 模块README
- **统一命名**: `README.md` (小写，各模块目录下)

## 文档分类

### 核心文档（根目录）
- `README.md` - 项目入口
- `TokenizerGraph_Detailed_Documentation.md` - 完整技术文档

### 指南文档（docs/guides/）
- 配置、实验、编码规范等标准指南

### 模块文档（src/*/README.md）
- 各模块的使用说明和API文档

### 专项文档（docs/）
- 代码扫描、集成指南等专项文档

### 归档文档（docs/archive/）
- 历史设计、规划、进度文档

## 文档链接更新

所有文档内的链接已更新到新位置：
- ✅ `docs/README.md` - 文档中心链接已更新
- ✅ `README.md` - 项目主文档链接已更新  
- ✅ `src/data/README.md` - 数据层文档链接已更新
- ✅ `docs/archive/README.md` - 归档索引已创建

## 维护说明

1. **新增文档**: 根据类型放入对应目录
2. **命名规范**: 遵循上述命名规范
3. **链接更新**: 移动文档后更新所有引用
4. **归档原则**: 历史文档移至 `docs/archive/`

---

**维护者**: 保持文档结构清晰，命名规范统一
