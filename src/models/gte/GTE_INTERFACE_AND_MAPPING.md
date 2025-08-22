# GTE 接口说明与层级对应关系

本文件说明：
- GTE 模型在本项目中的“接口（输入/输出）契约”；
- GTE 与本项目 BERT 的功能对应关系（在哪个层次上对齐）；
- GTE 与 HF `BertModel`（我们BERT底层实现）的层级映射/差异点；
- 方便在微调与未来预训练中，精确把握替换点与职责边界。

## 1. GTE 接口契约（基于上游实现/源码）

### 1.1 输入
- `input_ids`: `torch.LongTensor`，形状 `[batch_size, seq_len]`
- `attention_mask`: `torch.LongTensor`/`torch.FloatTensor`，形状 `[batch_size, seq_len]`（1=有效，0=padding）
- 设备：与模型同设备（CPU/GPU），dtype 符合要求（fp16 时需确保设备和编译环境支持）

### 1.2 输出
- `BaseModelOutputWithPooling`（与 HF 模型一致）：
  - `last_hidden_state`: `[batch_size, seq_len, hidden_size]`
  - `pooler_output`: `[batch_size, hidden_size]`（由首 token 线性 + Tanh 得到）
  - 可选 `hidden_states`/`attentions`

说明：根据本仓库包含的上游源码 `ref_gte_source_code/modeling.py::NewModel.forward`，默认返回的是标准的 Transformer 输出结构；并非“直接句向量”。若需要句向量，应当：
- 使用 `pooler_output`；或
- 对 `last_hidden_state` 进行 mean/cls 等池化（结合 `attention_mask`）。

### 1.3 运行时优化（来自上游配置/forward 参数）
- `unpad_inputs`（forward 参数，或 `config.unpad_inputs` 缺省）：启用基于 `attention_mask` 的 unpadding；
- `use_memory_efficient_attention`（来自 `config`）：可启用 xformers 的高效注意力；
- `torch_dtype`（`from_pretrained` 的参数）：可选半精度以节省显存；
以上字段均可在上游实现中找到对应入口，具体以模型实现为准。

## 2. 与本项目 BERT 的功能/层次对应

### 2.1 对齐层级
- 本项目 BERT 侧：`BertModel`（HF）输出 `last_hidden_state` `[B, L, H]`；我们在适配器层按 `attention_mask` 进行池化得到 `[B, H]`。
- GTE 侧（按上游源码）：`NewModel` 输出 `last_hidden_state` 与 `pooler_output`；我们可选择使用 `pooler_output` 或自行池化 `last_hidden_state` 得到 `[B, H]`。

因此，两者在“池化后句向量”层次对齐。上层（任务头/损失/指标）只消费 `[B, H]`。

### 2.2 维度与配置
- BERT-Small（默认）：`H=512`；GTE multilingual-base：`H=768`
- 统一处理方式：上层任务头只依赖 `get_hidden_size()`，无需感知差异

### 2.3 数据契约
- 都以 `input_ids` 与 `attention_mask` 为输入；
- pad id / mask 由 `VocabManager` + UDI 统一保证；
- 不在 encoder 层处理任务特定逻辑（如损失、指标）。

## 3. 与 HF BertModel 的层级对应

### 3.1 HF `BertModel`（当前 BERT 的底层实现）
- 典型输入：`input_ids`, `attention_mask`
- 典型输出：`BaseModelOutputWithPooling`
  - `last_hidden_state` `[B, L, H]`
  - `pooler_output` `[B, H]`（仅特定用法）
- 本项目中：我们取 `last_hidden_state`，再手动池化（避免 pooler 依赖）

### 3.2 GTE `NewModel`（参考源代码层级）
- `NewEmbeddings`：支持 `unpad_inputs`，在 embedding 阶段就做了 unpadding
- `NewEncoder`：可选 xformers 的 memory efficient attention
- `NewModel.forward`：支持 `inputs_embeds`/`input_ids`，返回 `BaseModelOutputWithPooling`
  - 若 `unpad_inputs=True`，内部使用 `BlockDiagonalMask` 等结构优化 attention 计算
- GTE 上层（AutoModel包装）通常直接提供句向量接口（项目中我们使用该句向量作为 encoder 输出）

### 3.3 映射关系（概念对齐）

| 本项目层级 | 依赖/实现 | GTE 对应 | 说明 |
|---|---|---|---|
| `BertModel` 输出 | `last_hidden_state` `[B, L, H]` | `NewModel`（内部序列表示） | 概念上同为 Transformer 编码序列 |
| 池化（项目实现） | mean/cls 池化 → `[B, H]` | GTE 直接返回 `[B, H]` | GTE 省略显式池化步骤，接口已完成 |
| 任务头与损失 | `TaskHandler` + 线性头 | 相同 | 与 encoder 无关、无需改动 |

## 4. 典型调用与对齐示例

### 4.1 BERT（项目）
```python
# 取序列表示
sequence_output = bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_dict=True
).last_hidden_state  # [B, L, H]

# 统一池化（mean）
mask = attention_mask.unsqueeze(-1).float()
pooled = (sequence_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
```

### 4.2 GTE（基于上游 `NewModel` 的标准调用）
```python
outputs = gte_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    return_dict=True,
    # 可选：unpad_inputs=True  # 若未在 config 中设置
)

# 方式1：使用 pooler_output（首 token，经线性+Tanh）
sentence_embeddings = outputs.pooler_output  # [B, H]

# 方式2：自行池化 last_hidden_state（推荐 mean with mask）
mask = attention_mask.unsqueeze(-1).float()
pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]
```

## 5. 微调与未来预训练的接口稳定性

- 微调：
  - 抽象 encoder（BERT/GTE）均提供统一的 `encode(...) -> [B, H]`，上层无需感知差异；
  - 任务头/损失/评估不变，端到端最小修改；
- 预训练（占位）：
  - 以 GTE 为 backbone 训练 token-level 任务（如 MLM）时，直接使用 `last_hidden_state`；
  - 可选做法：
    - A) 在包装层提供 `get_sequence_output(...) -> [B, L, H]`；
    - B) 单独实现“预训练专用 backbone 类”（避免影响微调路径）；
  - 该部分待确认任务定义/词表策略后再扩展。

## 6. 一致性与约束

- 输入/输出张量形状、dtype、设备需与上层一致；
- `attention_mask` 必须与 padding 约定一致（1=有效，0=padding）；
- GTE 的 `hidden_size` 与 BERT 的 `hidden_size` 可能不同，上层必须通过 `get_hidden_size()` 获取；
- 混合精度/高效注意力配置仅在 GTEEncoder 内部生效，不影响主流程；
- 抽象接口不承载任务逻辑，避免语义漂移。
