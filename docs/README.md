# 项目文档中心

> **最后更新**: 2025-10-30  
> **状态**: 文档整理完成

## 核心文档

### 技术文档

1. **[详细技术文档](../TokenizerGraph_Detailed_Documentation.md)** ⭐ - 完整的技术架构和使用指南
2. **[配置管理指南](guides/config_guide.md)** - 配置文件结构、参数管理、环境变量使用
3. **[实验指南](guides/experiment_guide.md)** - 实验设计、执行、评估的标准流程
4. **[编码规范](guides/coding_standards.md)** - 代码风格、命名约定、项目结构等编码标准
5. **[数据集导出指南](guides/dataset_export_guide.md)** - 数据集导出说明

### 模块文档

- **[数据层](../src/data/README.md)** - 数据加载和统一接口
- **[序列化算法](../src/algorithms/serializer/README.md)** - 图序列化方法
- **[BPE压缩](../src/algorithms/compression/README.md)** - BPE训练和编码

### 专项文档

- **[未使用代码扫描](unused_code_scan.md)** - 代码使用情况报告
- **[文档结构说明](DOCUMENTATION_STRUCTURE.md)** - 文档组织结构和命名规范
- **[项目状态（归档）](archive/current_project_status.md)** - 历史项目状态

## 文档使用指南

### 对于新加入项目的研究者

1. **快速开始**: 阅读根目录 [`README.md`](../README.md)
2. **了解架构**: 阅读 [`TokenizerGraph_Detailed_Documentation.md`](../TokenizerGraph_Detailed_Documentation.md)
3. **编码规范**: 阅读 [`guides/coding_standards.md`](guides/coding_standards.md)
4. **实验设计**: 阅读 [`guides/experiment_guide.md`](guides/experiment_guide.md)

### 对于日常开发

- **编写代码**: 参考 [`guides/coding_standards.md`](guides/coding_standards.md)
- **设计实验**: 参考 [`guides/experiment_guide.md`](guides/experiment_guide.md)
- **配置管理**: 参考 [`guides/config_guide.md`](guides/config_guide.md)
- **具体模块**: 查看对应模块的README

### 文档维护原则

- ✅ **零容忍错误**: 文档必须与代码严格一致
- ✅ **不确定即删除**: 不确定的内容直接删除，不要猜测
- ✅ **及时更新**: 代码变更后立即更新文档
- ✅ **版本标记**: 重要文档标注版本和更新日期

## 快速参考

### 关键原则

- **无静默失败**: 错误必须明确报告
- **数据真实性**: 禁止使用mock数据（除初期测试）
- **可重现性**: 固定种子、版本、环境
- **配置集中**: 单一配置源，避免分散

### 命名约定速查

- 类名：`PascalCase`
- 函数：`snake_case`
- 常量：`UPPER_SNAKE_CASE`
- 文件：`module_name.py`

### 实验流程速查

1. 验证配置
2. 设置环境
3. 准备数据
4. 训练模型
5. 评估结果
6. 生成报告

## 文档版本

| 文档 | 版本 | 最后更新 | 状态 |
|------|------|---------|------|
| 详细技术文档 | v2.0 | 2025-10-30 | ✅ 已更新 |
| 数据层文档 | v2.0 | 2025-10-30 | ✅ 已更新 |
| 序列化文档 | v2.0 | 2025-10-30 | ✅ 已更新 |
| BPE文档 | - | 2025-10-30 | ✅ 已验证 |
| 配置指南 | - | - | ✅ 保留 |
| 实验指南 | - | - | ✅ 保留 |

---

**维护者**: 发现文档错误请立即修正
