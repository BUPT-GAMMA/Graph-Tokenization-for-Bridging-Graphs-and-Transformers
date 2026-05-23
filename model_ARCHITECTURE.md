# TokenizerGraph 模型架构说明

本文档记录当前 release 范围内的模型构建路径。训练入口统一通过 `src/training/model_builder.py` 创建模型，模型本体由编码器、任务头和任务处理器组成。

## 入口

`build_task_model(config, udi, method, pretrained_dir=None, pretrain_exp_name=None, force_task_type=None, run_i=None)` 是当前训练流水线使用的模型构建入口。

预训练阶段由 `run_pretrain.py` 调用 `src/training/pretrain_pipeline.py`，并以 `force_task_type="mlm"` 创建 MLM 模型。微调阶段由 `run_finetune.py` 调用 `src/training/finetune_pipeline.py`，任务类型默认从 UDI 的数据集信息中推断。

## 构建流程

1. 解析任务类型。显式传入 `force_task_type` 时优先使用；否则从 `udi.get_dataset_task_type()` 获取。
2. 解析预训练权重路径。MLM 预训练任务固定不加载已有 checkpoint；微调任务按显式目录、预训练实验名、当前实验路径依次查找有效 checkpoint。
3. 从 `udi.get_vocab(method)` 读取词表，确定 `vocab_size` 和 `pad_token_id`。
4. 用 `src/models/unified_encoder.py` 中的 `create_encoder_from_config()` 创建 `BertEncoder` 或 `GTEEncoder`。
5. 用 `src/training/task_handler.py` 中的 `create_task_handler()` 创建任务处理器并确定输出维度。
6. 组装 `src/models/universal_model.py` 中的 `UniversalModel`。
7. 如果是微调且找到预训练 checkpoint，严格加载 checkpoint 中的 `encoder.*` 权重覆盖当前编码器。

## 主要组件

- `src/training/model_builder.py`: 模型构建、编码器配置、checkpoint 路径解析和预训练权重加载。
- `src/models/unified_encoder.py`: `BaseEncoder`、`BertEncoder`、`GTEEncoder` 和编码器工厂。
- `src/models/universal_model.py`: 统一模型容器，封装编码器、任务头和 pooling。
- `src/models/unified_task_head.py`: MLM、分类、回归等任务头。
- `src/training/task_handler.py`: 损失函数、输出后处理和指标计算。

## 编码器

`BertEncoder` 使用仓库内的 BERT 配置创建 HuggingFace `BertModel`，结构参数来自 `config/default_config.yml` 的 `bert.architecture`。

`GTEEncoder` 使用 `gte_model/` 中的本地配置和实现，对接 Alibaba-NLP/gte-multilingual-base。构建时会把项目词表大小和 `pad_token_id` 同步到 GTE embedding；`encoder.reset_weights=true` 时从配置随机初始化，反之加载 GTE 预训练权重后调整词表。

## 权重加载约束

微调加载预训练权重时，`_load_and_copy_pretrained_weights()` 会进行严格校验：

- checkpoint 目录必须包含 `config.bin` 和 `pytorch_model.bin`。
- checkpoint 中必须存在与当前编码器匹配的 `encoder.*` 权重。
- 词嵌入行数必须等于当前词表大小。
- 若双方都有绝对位置嵌入，位置嵌入长度必须一致。
- checkpoint 的编码器参数集合必须与当前编码器参数集合完全一致。

这些约束用于避免错误 checkpoint、错误词表或错误模型结构被静默加载。

## 数据流

MLM 预训练：

```text
input_ids, attention_mask
  -> encoder.get_sequence_output()
  -> MLM task head
  -> token-level logits [batch, seq_len, vocab_size]
```

微调：

```text
input_ids, attention_mask
  -> encoder sequence output
  -> pooling
  -> task head
  -> task prediction
```

## 扩展点

新增编码器时，在 `src/models/unified_encoder.py` 中实现 `BaseEncoder` 接口，并在 `create_encoder_from_config()` 中添加分发逻辑。

新增任务类型时，在 `src/training/task_handler.py` 中定义任务处理逻辑，并在 `src/models/unified_task_head.py` 中补充对应任务头。
