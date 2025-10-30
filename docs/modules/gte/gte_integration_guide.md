## GTE 集成与词表对齐指南（微调与重新预训练）

本指南说明如何用“自有 token id 序列”对接 GTE（Alibaba-NLP/gte-multilingual-base），并分别给出：
- 用自有 ID 直接微调 GTE 预训练权重
- 从零初始化（或在加载后重置全部参数）重新预训练
- 可选的 `inputs_embeds` 方案

同时明确与 GTE 自带 tokenizer 的职责边界：你可以完全不使用其 tokenizer，而是沿用自己的 `VocabManager` 产出的 `input_ids/attention_mask`，只需确保词表大小与 `pad_token_id` 对齐即可。

---

### 1) 关键信息与不变量

- GTE 默认配置（见 `gte_model/config.json`）：
  - `vocab_size=250048`
  - `pad_token_id=1`
  - `<mask>` 的 id 为 250001（仅在 MLM 任务时由数据侧使用）
- 模型在 `gte_model/modeling.py` 中使用 `config.pad_token_id` 作为 `nn.Embedding(..., padding_idx=...)` 的零行；注意与输入 `attention_mask` 一致。
- 你的流水线可以继续使用 `VocabManager` 来打包 `input_ids/attention_mask`（ID 级别的截断/pad/特殊符号），无需 GTE 自带 tokenizer。

---

### 2) 用自有 ID 微调 GTE 预训练权重（推荐）

场景：保留 Transformer 编码器权重，仅让模型接受你的 id 空间与 pad 协议。

步骤：

```python
import torch
from transformers import AutoModel

# 强制使用 GPU
assert torch.cuda.is_available(), "必须使用GPU执行模型相关计算"
device = torch.device("cuda")

new_vocab_size = YOUR_VOCAB_SIZE   # 例如 len(vocab_manager)
pad_id = YOUR_PAD_ID               # 例如 0

model = AutoModel.from_pretrained(
    "Alibaba-NLP/gte-multilingual-base",
    trust_remote_code=True,
).to(device)

# 1) 调整词表大小（变大会随机初始化新行；变小会截断）
model.resize_token_embeddings(new_vocab_size)

# 2) 对齐 pad 协议并确保 pad 行为零
model.config.pad_token_id = pad_id
emb = model.get_input_embeddings()
emb.padding_idx = pad_id
with torch.no_grad():
    emb.weight[pad_id].zero_()

model.eval()

# 3) 用你的 VocabManager 打包输入
batch = vocab_manager.encode_batch(token_id_sequences, add_special_tokens=True, max_length=8192)
batch = {k: v.to(device) for k, v in batch.items()}

# 4) 前向（GTE 支持 unpad_inputs）
with torch.no_grad():
    outputs = model(**batch, unpad_inputs=True)

# 5) 句向量（CLS 位置），并归一化
import torch.nn.functional as F
embeddings = outputs.last_hidden_state[:, 0]
embeddings = F.normalize(embeddings, p=2, dim=1)
```

必做校验（fail-fast）：

```python
assert batch['input_ids'].dtype == torch.long
assert int(batch['input_ids'].max()) < new_vocab_size
```

说明：
- 不必使用 GTE 的 tokenizer；你已有的 ID 流水线即可。
- 只在做 MLM 时才需要自定义 `mask_token_id` 与 `labels`（非 mask 位置为 -100）。

---

### 2.1) 模型内部使用了哪些“特殊 token/词表”信息？

与源码对应的关键点如下（只列“模型会读/依赖”的部分）：

```284:305:gte_model/modeling.py
class NewEmbeddings(nn.Module):
    def __init__(self, config: NewConfig):
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type == 'absolute':
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
```

- **config.vocab_size（模型内部使用）**：决定 `word_embeddings` 的行数；与 `input_ids` 的取值上界必须一致。
- **config.pad_token_id（模型内部使用）**：作为 `word_embeddings.padding_idx`（以及 absolute 位置嵌入时的 `padding_idx`），该行权重会被置零；需与你的 pad 协议一致。
- **position_embedding_type / rope_scaling / max_position_embeddings（模型内部使用）**：位置编码相关；与 tokenizer 无直接绑定。
- **type_vocab_size（模型内部使用）**：>0 时会创建 `token_type_embeddings` 并在前向中相加；若不需要 token type，建议在“从配置构造模型”时设为 0（已构造的实例仅改 config 值不会移除该层）。

前向中的注意力 mask 与去 padding 逻辑：

```333:418:gte_model/modeling.py
def forward(self, unpad_inputs: bool, input_ids=None, attention_mask=None, ...):
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
        ...
    if unpad_inputs:
        attention_mask_bool = attention_mask.bool()
        if length is None:
            length = attention_mask.sum(-1).tolist()
    ...
    if inputs_embeds is None:
        if unpad_inputs:
            input_ids = input_ids[attention_mask_bool].unsqueeze(0)
        inputs_embeds = self.word_embeddings(input_ids)
```

- **attention_mask（模型前向使用）**：用于生成 `attention_bias` 或做 unpadding gather；需与 pad 位置一致（pad 处为 0）。
- **mask_token_id（模型不使用）**：仅在 MLM 训练的数据侧生效（替换输入与构造 `labels`）。
- **[CLS]/[SEP]/bos/eos（模型不使用）**：属于 tokenizer/数据侧约定；模型只按索引取嵌入，不关心语义。常见做法是将“首位”作为聚合位（CLS），再取 `last_hidden_state[:, 0]`。

---

### 3) 从零开始重新预训练（随机初始化）

场景：完全对齐你的词表与 pad 协议，从头训练权重。

```python
import torch
from transformers import AutoModel
from gte_model.configuration import NewConfig

assert torch.cuda.is_available(), "必须使用GPU执行模型相关计算"
device = torch.device("cuda")

config = NewConfig(
    vocab_size=new_vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=8192,
    position_embedding_type="rope",
    unpad_inputs=True,
    use_memory_efficient_attention=True,
    attention_probs_dropout_prob=0.0,
    hidden_dropout_prob=0.1,
    pad_token_id=pad_id,
)

# 从配置构造随机初始化模型
model = AutoModel.from_config(config).to(device)

# 置零 pad 行
emb = model.get_input_embeddings()
emb.padding_idx = pad_id
with torch.no_grad():
    emb.weight[pad_id].zero_()
```

做 MLM 预训练（可选）：

```python
from transformers import AutoModelForMaskedLM

mlm_model = AutoModelForMaskedLM.from_config(config).to(device)

# 数据侧：按你的协议将部分 token 替换为 YOUR_MASK_ID，并构造 labels（非 mask 位置置 -100）
# 注意：如果使用 GTE 的 tokenizer 协议以外的 mask id，需确保 collator 与 labels 一致
```

---

### 4) “加载→改 embedding 大小→重置全部参数→预训练” 可以吗？

可以。常见做法有两种：

方案 A（更直接）：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
model.resize_token_embeddings(new_vocab_size)
model.config.pad_token_id = pad_id
emb = model.get_input_embeddings()
emb.padding_idx = pad_id

# 重新初始化全部权重（正确方式）
if hasattr(model, "init_weights"):
    model.init_weights()               # transformers.PreTrainedModel 提供
else:
    model.apply(model._init_weights)   # 退化到逐模块初始化

with torch.no_grad():
    emb.weight[pad_id].zero_()
```

说明：
- `init_weights()` 会调用内部 `_init_weights`，并按 `config.initializer_range` 正确初始化；
- 若使用 `AutoModelForMaskedLM`，先 `resize_token_embeddings(new_vocab_size)`，再 `model.init_weights()`，最后如需权重共享可调用 `model.tie_weights()`（注意：`tie_weights()` 只适用于带 LM 头的模型，`AutoModel` 基类无此需求）。

方案 B（更简洁）：直接 `AutoModel.from_config(NewConfig(...))`，无需载入预训练后再重置。

---

### 5) 可选：只用 `inputs_embeds`（绕开词表与特殊符号）

适合已有外部嵌入表或想完全自定义嵌入行为：

```python
E = torch.nn.Embedding(new_vocab_size, model.config.hidden_size).to(device)
input_ids = batch['input_ids'].to(device)
inputs_embeds = E(input_ids)

outputs = model(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'].to(device))
```

限制：无法直接做 MLM（需要输出维度与词表一致的头部）。

---

### 6) 训练与工程注意事项

- 始终使用 GPU；无 GPU 直接失败（fail-fast）。
- 严格断言 ID 上界、小于 `vocab_size`；`attention_mask` 与 pad 位置一致。
- 若使用 `unpad_inputs=True` 且启用 memory-efficient attention，注意批内长度统计的正确性（GTE 已在 forward 内部兼容）。
- CLS 向量取法与官方一致：`outputs.last_hidden_state[:, 0]`，如需维度裁剪自行切片，然后 `F.normalize(..., p=2, dim=1)`。

---

### 8) 职责边界与需要改动的地方（清单）

- 模型内部会“读取/依赖”的配置与张量：
  - **config.vocab_size**：需匹配你的 ID 空间（否则会越界）。
  - **config.pad_token_id**：用于 `padding_idx`；需与你的 pad 协议一致，并在修改后将 `emb.weight[pad_id]` 置零。
  - **position_embedding_type / rope 参数**：训练策略层面决定，与你的 tokenizer 无直接耦合。
  - **type_vocab_size**：是否启用 token type 嵌入；若想禁用，需在“构造模型”时设为 0。
  - **attention_mask**：前向必须与 pad 位置一致（token=1，pad=0）。

- tokenizer/数据侧负责的事项：
  - 文本切分/子词规则、字符串→id 的映射；
  - 添加/移除特殊符号（如 [CLS]/[SEP]）、截断、padding、构造 `attention_mask`；
  - MLM 时的 `<mask>` 插入与 `labels`（非 mask 位置为 -100）。

- 二者之间的“接口契约”：
  - `input_ids` ∈ `[0, config.vocab_size-1]`；
  - `attention_mask` 与 pad 位置一致（形状 `[bs, seq]`，dtype `bool/long/float` 均可；unpadding 会调用 `.bool()`）；
  - 模型不检查“某个 id 是否代表 CLS/SEP/UNK/…”，是否插入由你决定；若使用 CLS first pooling，请确保首位的确是聚合位。

- 你需要改什么（按目标场景）：
  - 微调：`resize_token_embeddings(new_vocab_size)` → 设置 `config.pad_token_id` 与 `emb.padding_idx` → 置零 pad 行 → 用你的 `VocabManager` 打包输入（确保 pad/attention_mask 一致）。
  - 重新预训练（随机初始化）：用 `NewConfig(vocab_size=..., pad_token_id=..., type_vocab_size=0/1...)`，`AutoModel.from_config(config)` 构造后置零 pad 行；若做 MLM，改用 `AutoModelForMaskedLM.from_config(config)` 并自定义 collator。
  - 载入后重置：按“方案 A”先对齐词表与 pad，再 `model.init_weights()` 重置全参；如为 MLM 模型，必要时 `model.tie_weights()`。

---

### 7) 最小可运行微调骨架

```python
import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available()
device = torch.device("cuda")

new_vocab_size = YOUR_VOCAB_SIZE
pad_id = YOUR_PAD_ID

model = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
model.resize_token_embeddings(new_vocab_size)
model.config.pad_token_id = pad_id
emb = model.get_input_embeddings()
emb.padding_idx = pad_id
with torch.no_grad():
    emb.weight[pad_id].zero_()

head = nn.Linear(model.config.hidden_size, 1).to(device)

batch = vocab_manager.encode_batch(token_id_sequences, add_special_tokens=True, max_length=8192)
batch = {k: v.to(device) for k, v in batch.items()}

outputs = model(**batch, unpad_inputs=True)
cls = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
pred = head(cls)
loss = F.mse_loss(pred.squeeze(-1), labels.to(device))
loss.backward()
# optimizer.step(), scheduler.step() ...
```

---

如需进一步封装 `Adapter` 以适配你现有的 `VocabManager` 到 GTE 的前向签名，可在 `src/models/gte/` 下新增包装器，内部仅做 pad 对齐与张量打包，遵循本指南的约束与断言即可。


