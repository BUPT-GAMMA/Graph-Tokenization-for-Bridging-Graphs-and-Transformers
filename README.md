# TokenizerGraph 项目文档

> **最后更新**: 2025-10-30  
> **项目状态**: 论文投稿完成，代码整理中

## 快速开始

### 核心文档

1. **[详细技术文档](TokenizerGraph_Detailed_Documentation.md)** - 完整的技术架构和使用指南
2. **[配置管理指南](docs/guides/config_guide.md)** - 配置文件结构和使用
3. **[实验指南](docs/guides/experiment_guide.md)** - 实验设计和执行规范
4. **[编码规范](docs/guides/coding_standards.md)** - 代码风格和规范

### 模块文档

- **数据层**: [`src/data/README.md`](../src/data/README.md)
- **序列化算法**: [`src/algorithms/serializer/README.md`](../src/algorithms/serializer/README.md)
- **BPE压缩**: [`src/algorithms/compression/README.md`](../src/algorithms/compression/README.md)

## 项目结构

```
tokenizerGraph/
├── config.py                    # 统一配置管理
├── run_pretrain.py             # 预训练入口
├── run_finetune.py             # 微调入口
├── prepare_data_new.py          # 数据预处理
├── src/
│   ├── data/                   # 数据层
│   ├── algorithms/             # 算法层（序列化、BPE）
│   ├── models/                 # 模型层（BERT、GTE）
│   └── training/               # 训练层
└── docs/                       # 文档中心
```

## 使用流程

### 1. 数据预处理

```bash
python prepare_data_new.py \
    --dataset qm9test \
    --method feuler \
    --bpe_num_merges 2000
```

### 2. 预训练

```bash
python run_pretrain.py \
    --dataset qm9test \
    --method feuler \
    --epochs 100 \
    --batch_size 256
```

### 3. 微调

```bash
python run_finetune.py \
    --dataset qm9test \
    --method feuler \
    --task regression \
    --target_property homo
```

## 文档维护状态

| 文档 | 状态 | 最后更新 |
|------|------|---------|
| 详细技术文档 | ✅ 已更新 | 2025-10-30 |
| 数据层文档 | ✅ 已更新 | 2025-10-30 |
| 序列化文档 | ✅ 已更新 | 2025-10-30 |
| BPE文档 | ✅ 已验证 | 2025-10-30 |
| 配置指南 | ✅ 保留 | - |
| 实验指南 | ✅ 保留 | - |

## 代码整理状态

- ✅ 核心文档已更新并验证
- ✅ 未使用代码扫描完成（见 [`docs/unused_code_scan.md`](docs/unused_code_scan.md)）
- ✅ 文档结构已重构（统一命名和位置）
- ⏳ 代码注释更新（待完成）

## 相关资源

- **未使用代码扫描**: [`docs/unused_code_scan.md`](docs/unused_code_scan.md)
- **项目状态（归档）**: [`docs/archive/current_project_status.md`](docs/archive/current_project_status.md)

---

**维护者**: 发现文档错误请立即修正

