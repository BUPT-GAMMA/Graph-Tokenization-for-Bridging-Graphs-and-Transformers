# 文档结构说明

本文件记录当前 release 范围内保留的稳定文档入口。过程性报告、实验输出、数据库、编译产物和未引用图片不属于发布文档。

## 入口文档

- `README.md`: 英文项目入口。
- `README_zh.md`: 中文项目入口。
- `docs/README.md`: 文档索引。

## 指南文档

- `docs/guides/config_guide.md`: 配置系统与参数说明。
- `docs/guides/experiment_guide.md`: 实验运行与训练流程说明。
- `docs/guides/hyperparameter_search.md`: Optuna 调参工作流、脚本入口和 smoke-test 路径。
- `docs/reproducibility/environment-setup.md`: 环境准备。
- `docs/reproducibility/paper-dataset-cold-start-guide.md`: 数据集冷启动与复现说明。

## 保留的专项说明

- `docs/datasets_overview.md`: 数据集、指标、存储格式和 loader 入口总览。
- `docs/gte_integration_guide.md`: GTE 与自有 token ID 空间的对接说明。
- `docs/bpe/BPE_USAGE_GUIDE.md`: BPE 使用方式。
- `docs/bpe/README.md`: BPE 模块概览。
- `docs/bpe/bpe_transform_rework.md`: BPE transform 与多次序列化的设计记录。
- `model_ARCHITECTURE.md`: 统一模型构建与预训练权重加载说明。
- `final/README.md`: 保留的实验数据加载器说明。

## 模块文档

- `src/data/README.md`: 数据层和 UDI 说明。
- `src/algorithms/serializer/README.md`: 图序列化算法说明。
- `src/algorithms/compression/README.md`: BPE 引擎实现说明。
- `src/models/README.md`: 模型层结构说明。
- `src/models/bert/README.md`: BERT 相关组件说明。
- `src/training/README.md`: 预训练、微调与评估流程说明。
- `scripts/dataset_conversion/README.md`: 数据转换与检查脚本说明。

## 维护原则

新增文档应优先链接到 `docs/README.md`。只保留对复现、代码理解或论文结果解释有长期价值的文档；一次性过程记录、未使用图片、数据库、日志和二进制实验输出不应纳入 release。
