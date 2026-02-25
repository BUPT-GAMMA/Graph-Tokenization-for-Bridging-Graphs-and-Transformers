# 文档整理与翻译 — 中途进度记录

**当前分支**: `release`
**日期**: 2025-02-25

---

## ✅ 已完成

### 1. 文档归档清理
- 创建 `_docs_to_review/` 文件夹，移入 11 个冗余/AI痕迹重的文档
- 每个文件的来源和归档原因见 `_docs_to_review/README.md`
- **用户确认**：数据集统计信息文档已整合完成
- **用户确认**：数据集处理脚本文档已整合完成

### 2. 新建模块 README
- `src/models/README.md` — 模型模块文档（架构图、组件表、用法示例、任务类型）
- `src/training/README.md` — 训练模块文档（工作流、组件表、预训练/微调用法、指标说明）

### 3. 根 README 分支标注
- `README.md` 顶部添加醒目 blockquote 标注 release/dev 分支
- `README_zh.md` 同步添加中文分支说明

### 4. docs/README.md 索引更新
- 添加 Models、BERT Encoder、Training 三个链接

### 5. 已翻译的代码文件（之前会话完成）
以下文件的中文注释/docstring 已在之前的会话中翻译为英文（大量文件在 src/algorithms/, src/data/, src/utils/ 下）。
完整变更列表见 `git status`，共约 70+ 个文件被修改。

`src/models/__init__.py` 已翻译（双语 docstring 格式）。

---

## 🔲 待完成

### A. src/models/ 中文注释翻译（13 个文件仍含中文）

**翻译规则**（用户要求）：
1. **关键注释保留中文**（或中英双语）
2. **函数 docstring 保留中文**（英文+中文双语格式）
3. 之前**完全删除中文的 docstring 需要补回中文**

**待翻译文件**（按中文行数从少到多）：
| 文件 | 中文行数 | 备注 |
|------|---------|------|
| `src/models/gte/__init__.py` | 2 | 极小 |
| `src/models/__init__.py` | 6 | 已双语，可能需微调 |
| `src/models/utils/pooling.py` | 6 | 小文件 |
| `src/models/aggregators/variant_weighting.py` | 10 | 小文件 |
| `src/models/bert/__init__.py` | 16 | |
| `src/models/bert/config.py` | 18 | |
| `src/models/unified_encoder.py` | 27 | |
| `src/models/bert/transforms.py` | 42 | |
| `src/models/model_factory.py` | 42 | |
| `src/models/unified_task_head.py` | 60 | |
| `src/models/universal_model.py` | 68 | |
| `src/models/bert/vocab_manager.py` | 96 | 较大 |
| `src/models/bert/data.py` | 166 | 最大 |

### B. 检查之前已翻译文件，补回中文 docstring
- 检查 `src/models/__init__.py`（已是双语格式 ✅）
- 检查其他之前翻译过的文件（src/algorithms/, src/data/, src/utils/ 等），确认 docstring 是否需要补中文

### C. 同步 release 和 dev 分支
1. 在 release 分支提交所有变更
2. 切换到 dev 分支，合并或 cherry-pick release 的文档变更
3. 确认两个分支的 README 都正确标注了对方分支
4. 推送两个分支

### D. _docs_to_review/ 最终清理
- 等待用户 review 后删除整个文件夹

---

## 📝 注意事项

- **翻译风格**：docstring 采用 `English description\n中文描述` 双语格式
- **不修改代码逻辑**：只改注释和文档
- **print/logger 中的中文**：保持不变（运行时输出，不影响可读性）
- **assert 消息中的中文**：保持不变（内部调试用）
