# BPE for Images 项目完整文档

> **最后更新**: 2025-10-30  
> **项目状态**: 初步完成，准备独立部署

---

## 目录

1. [项目背景与初衷](#项目背景与初衷)
2. [实验设计](#实验设计)
3. [实现架构](#实现架构)
4. [当前功能](#当前功能)
5. [序列化策略](#序列化策略)
6. [未来计划](#未来计划)
7. [技术细节](#技术细节)
8. [使用指南](#使用指南)

---

## 项目背景与初衷

### 为什么做这个项目？

本项目来源于主项目 **TokenizerGraph** 的核心研究思想。TokenizerGraph 是一个图序列化框架，通过将图结构转换为序列数据，然后使用 BPE 压缩和 Transformer 模型进行学习。

**核心研究问题：**
- 图数据能否通过序列化+BPE+Transformer的方式有效建模？
- BPE 压缩是否能发现有意义的结构模式？
- Transformer 能否处理这种序列化的图/图像数据？

### 从图到图像的自然延伸

在 TokenizerGraph 项目中，我们已经验证了：
1. **图序列化算法** - 将图结构转换为序列
2. **BPE 压缩** - 发现子结构模式
3. **Transformer 学习** - 从序列中学习表示

**自然延伸：图像也是特殊类型的图**
- 图像可以看作是一种规整的图（像素是节点，邻接关系是边）
- 如果将图像展平为序列，能否应用相同的 BPE+Transformer 方法？
- 这是一个更简单、更可控的实验场景，便于验证核心思想

### 实验目标

**主要研究问题：**
1. **BPE 在图像数据上是否有效？**
   - 对比直接 Transformer vs BPE+Transformer
   - 验证 BPE 是否能够发现图像的局部模式（如边缘、纹理）

2. **序列化策略对性能的影响？**
   - 不同的展平顺序（行优先、列优先、蛇形、对角线）如何影响模型性能？
   - 是否某些序列化策略更适合 Transformer 建模？

3. **Transformer vs 传统方法？**
   - 在简单图像任务（MNIST）上，Transformer 与 CNN/MLP 的性能对比
   - 序列长度、参数量、训练效率的权衡

---

## 实验设计

### 数据集

**MNIST 手写数字数据集**
- 训练集：60,000 张图像（28×28 灰度图）
- 测试集：10,000 张图像
- 类别数：10（数字 0-9）

**数据格式：**
- 原始：28×28 灰度图像（像素值 0-255）
- 展平：784 长度的整数序列（0-255）
- BPE 压缩后：约 140 长度（压缩率 ~18%）

### 对比方案

#### 1. Baseline 模型

**MLP (多层感知机)**
- 架构：784 → 512 → 256 → 10
- 用途：最简单的全连接网络，验证数据是否线性可分

**LeNet-5 (CNN)**
- 架构：Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10)
- 用途：经典卷积网络，在 MNIST 上表现优异（~99%），作为性能上限参考

#### 2. Transformer 方案

**Transformer (灰度值直接作为 token)**
- 输入：784 长度的灰度值序列（vocab_size=256）
- 架构：BERT-Small（d_model=256, n_layers=4, n_heads=4）
- 用途：验证 Transformer 能否直接从展平序列学习

**BPE+Transformer**
- BPE 压缩：将 784 长度压缩到 ~140（压缩率 ~18%）
- 输入：压缩后的序列（vocab_size=256+200=456）
- 架构：同 Transformer
- 用途：验证 BPE 压缩是否有助于提升性能

#### 3. Transformer 架构变体

**BERT 架构**
- 标准 BERT-Small 配置
- CLS token pooling
- 从头训练（随机初始化）

**GTE 架构**（可选）
- 使用 GTE 编码器
- 支持对比实验

### 序列化策略

**核心创新点：不同的展平策略**

目前实现了四种展平策略：

1. **row（行优先）** - 默认策略
   - 从左到右、从上到下逐行扫描
   - 最自然的展平方式

2. **col（列优先）**
   - 从上到下、从左到右逐列扫描
   - 测试不同的空间信息组织方式

3. **snake（蛇形扫描）**
   - 奇数行反向，偶数行正向
   - 模拟图像处理中的 zigzag 扫描

4. **diag（对角线扫描）** - 当前占位，后续实现
   - 类似 JPEG 的之字形对角线扫描
   - 优先捕获低频信息（对角线方向）

**每种策略应用场景：**
- MLP：所有策略（验证策略对简单模型的影响）
- Transformer：所有策略（验证策略对序列模型的影响）
- BPE+Transformer：所有策略（验证策略对压缩效果的影响）

### 评估指标

1. **准确率**
   - 测试集准确率（主要指标）
   - 验证集准确率（训练过程监控）

2. **训练效率**
   - 每个 epoch 的训练时间
   - 总训练时间

3. **模型复杂度**
   - 总参数量
   - 编码器参数量
   - 分类头参数量

4. **序列压缩**
   - BPE 压缩率
   - 平均序列长度变化

---

## 实现架构

### 项目结构

```
bpe_for_images/
├── bpe_config.py                    # 统一配置管理
├── bpe_data/                        # 数据模块
│   ├── __init__.py
│   ├── mnist_loader.py              # MNIST数据加载器（支持多种展平策略）
│   └── bpe_processor.py             # BPE处理器（C++后端）
├── bpe/                             # BPE引擎（C++后端，独立模块）
│   ├── bpe_engine.py                # BPE引擎接口
│   ├── cpp_bpe_backend.py           # C++后端包装
│   ├── _cpp_bpe.cpp                 # C++源码
│   └── setup.py                      # 构建脚本
├── models/                          # 模型模块
│   ├── __init__.py
│   ├── mlp_classifier.py            # MLP分类器
│   ├── lenet.py                      # LeNet-5
│   ├── transformer_classifier.py    # Transformer分类器
│   └── encoder.py                    # 轻量BERT编码器
├── utils/                            # 工具模块
│   └── logger.py                     # 日志工具
├── training_utils.py                 # 训练工具函数
├── train_mlp.py                      # 训练MLP（支持策略）
├── train_lenet.py                    # 训练LeNet
├── train_transformer.py             # 训练Transformer（支持策略）
├── train_bpe.py                      # 训练BPE模型（支持策略）
├── train_bpe_transformer.py          # 训练BPE+Transformer（支持策略）
├── evaluate.py                       # 统一评估
├── compare_results.py                # 结果对比
├── visualize_results.py              # 结果可视化（按策略分组）
├── saved_bpe_models/                 # BPE模型保存目录（按策略分别保存）
├── checkpoints/                      # 模型检查点
├── results/                          # 实验结果（JSON/CSV/PNG）
├── run_all_experiments.sh            # 一键运行所有实验
├── README.md                         # 快速开始指南
└── PROJECT_DOCUMENTATION.md         # 本文档（完整项目文档）
```

### 设计原则

#### 1. 完全独立
- 所有依赖代码都已复制到本项目
- 不依赖主项目（除了验证性引用）
- 可以独立部署和运行

#### 2. 模块化设计
- **数据模块** (`bpe_data/`) - 数据加载和预处理
- **模型模块** (`models/`) - 模型定义
- **BPE模块** (`bpe/`) - BPE引擎（C++后端）
- **训练模块** (`training_utils.py`) - 通用训练逻辑
- **评估模块** (`evaluate.py`, `compare_results.py`) - 结果评估和对比

#### 3. 策略驱动
- 所有训练脚本支持 `--flatten_strategy` 参数
- 模型名称包含策略信息（如 `mlp_row`, `transformer_bert_snake`）
- 结果文件按策略分别保存

#### 4. 科研友好
- 完整的训练日志
- 每个 epoch 的详细记录
- 自动保存检查点和结果
- 丰富的可视化图表

---

## 当前功能

### ✅ 已实现功能

#### 1. 数据加载与预处理

**MNIST 数据加载器** (`bpe_data/mnist_loader.py`)
- ✅ 支持三种数据格式：`image`（CNN）、`flatten`（MLP）、`sequence`（Transformer）
- ✅ 支持四种展平策略：`row`, `col`, `snake`, `diag`（diag 占位，待实现）
- ✅ 自动划分训练/验证/测试集
- ✅ 支持原始数据加载（用于 BPE 训练）

**展平策略实现** (`_build_flatten_index`)
- ✅ `row`: 行优先扫描（从左到右、从上到下）
- ✅ `col`: 列优先扫描（从上到下、从左到右）
- ✅ `snake`: 蛇形扫描（奇数行反向）
- ⚠️ `diag`: 对角线扫描（占位实现，待完善）

#### 2. BPE 处理

**BPE 处理器** (`bpe_data/bpe_processor.py`)
- ✅ 使用 C++ 后端 BPEEngine（高性能）
- ✅ 支持训练和编码
- ✅ 模型保存和加载
- ✅ 压缩统计信息

**BPE 引擎** (`bpe/bpe_engine.py`)
- ✅ C++ 后端实现（`bpe/_cpp_bpe.cpp`）
- ✅ 训练：使用 C++ 后端，速度提升 ~40 倍
- ✅ 编码：使用 C++ 后端，批量处理高效
- ✅ 支持保存和加载 codebook

#### 3. 模型实现

**MLP 分类器** (`models/mlp_classifier.py`)
- ✅ 可配置的隐藏层结构
- ✅ 支持多种激活函数
- ✅ Dropout 正则化

**LeNet-5** (`models/lenet.py`)
- ✅ 经典 CNN 架构
- ✅ 在 MNIST 上表现优异（~99%）

**Transformer 分类器** (`models/transformer_classifier.py`)
- ✅ 基于 BERT-Small 架构
- ✅ 支持 BERT 和 GTE 两种编码器
- ✅ CLS token pooling
- ✅ 可配置的分类头

#### 4. 训练脚本

**训练脚本**（支持 `--flatten_strategy` 参数）
- ✅ `train_mlp.py` - MLP 训练（支持所有策略）
- ✅ `train_lenet.py` - LeNet 训练
- ✅ `train_transformer.py` - Transformer 训练（支持所有策略）
- ✅ `train_bpe.py` - BPE 模型训练（支持所有策略）
- ✅ `train_bpe_transformer.py` - BPE+Transformer 训练（支持所有策略）

**训练特性**
- ✅ 统一的训练循环（`training_utils.py`）
- ✅ 自动保存最佳模型
- ✅ 每个 epoch 记录详细指标
- ✅ 学习率调度（CosineAnnealing）
- ✅ 强制使用 CUDA（GPU 必需）

#### 5. 评估与可视化

**评估脚本**
- ✅ `evaluate.py` - 统一评估所有模型
- ✅ `compare_results.py` - 生成对比表格（CSV）

**可视化脚本** (`visualize_results.py`)
- ✅ 训练曲线聚合图（Train/Val Accuracy 分别叠加）
- ✅ 准确率对比柱状图
- ✅ 效率对比散点图（参数量、训练时间 vs 准确率）
- ✅ 单模型详细曲线（`individual_curves/`）
- ⚠️ **待实现**：按策略分组的可视化

---

## 序列化策略

### 当前实现的策略

#### 1. row（行优先）✅

**扫描方式：**
```
[0,  1,  2, ..., 27]
[28, 29, 30, ..., 55]
...
[756, 757, ..., 783]
```

**实现：**
```python
idxs = [r * w + c for r in range(h) for c in range(w)]
```

**特点：**
- 最自然的展平方式
- 保持行的连续性
- 适合处理水平方向的模式

#### 2. col（列优先）✅

**扫描方式：**
```
[0,  28, 56, ..., 756]
[1,  29, 57, ..., 757]
...
[27, 55, 83, ..., 783]
```

**实现：**
```python
idxs = [r * w + c for c in range(w) for r in range(h)]
```

**特点：**
- 保持列的连续性
- 适合处理垂直方向的模式
- 与行优先形成对比

#### 3. snake（蛇形扫描）✅

**扫描方式：**
```
[0,  1,  2, ..., 27]      ← 正向
[55, 54, 53, ..., 28]      ← 反向
[56, 57, 58, ..., 83]      ← 正向
...
```

**实现：**
```python
for r in range(h):
    row = [r * w + c for c in range(w)]
    if r % 2 == 1:
        row.reverse()  # 奇数行反向
    idxs.extend(row)
```

**特点：**
- 模拟图像处理中的 zigzag 扫描
- 减少行与行之间的跳变
- 更适合捕获局部连续性

#### 4. diag（对角线扫描）⚠️ 待实现

**设计目标：类似 JPEG 的之字形对角线扫描**

**预期扫描方式（类似 JPEG Zigzag）：**
```
[0]                         # 对角线 0
[1,  28]                    # 对角线 1 (正向：左上->右下)
[56, 29,  2]                # 对角线 2 (反向：右下->左上)
[3,  30,  57, 84]           # 对角线 3 (正向)
...
```

**当前状态：**
- ✅ 基本框架已实现（代码中有占位实现）
- ⚠️ **但逻辑需要完善，待后续实现**
- 当前实现：简单按对角线分组，之字形方向逻辑待优化

**后续实现计划：**

对角线扫描应该按照以下方式实现：
1. **收集所有对角线**：按 r+c = const 分组
2. **对角线内部排序**：每条对角线按反对角线方向（从左上到右下，或从右下到左上）
3. **之字形方向**：
   - 偶数对角线（k % 2 == 0）：从左上到右下
   - 奇数对角线（k % 2 == 1）：从右下到左上（需要反转）

**技术要点：**
- 需要正确处理边界情况（对角线长度变化）
- 确保每个像素都被访问且只访问一次
- 对角线内部的扫描顺序需要仔细设计
- 需要验证与 JPEG Zigzag 扫描的一致性

**预期特点：**
- 优先捕获低频信息（对角线方向，类似 JPEG DCT）
- 类似 JPEG 的 zigzag 扫描模式
- 可能更适合 Transformer 建模（更好的空间相关性）
- 可能发现更多有意义的空间模式

**优先级：**
- 🔴 **高优先级**：对角线扫描策略是实现序列化策略对比的关键
- 📅 **计划时间**：项目迁移后第一周完成

### 策略对比实验

**实验设计：**
- 对每种策略运行 MLP、Transformer、BPE+Transformer
- 对比不同策略的性能差异
- 分析策略对 BPE 压缩率的影响

**预期发现：**
- 不同策略可能影响 Transformer 的建模效果
- 某些策略可能更适合 BPE 发现模式
- 对角线扫描可能捕获更多空间相关性

---

## 未来计划

### 短期计划（项目迁移后 1-2周）

#### 1. 实现对角线扫描策略 🔴 **高优先级**

**目标：完整实现类似 JPEG 的对角线扫描**

**当前状态：**
- 代码框架已存在（`bpe_data/mnist_loader.py` 中的 `_build_flatten_index`）
- 但实现逻辑需要完善和验证

**技术细节：**
1. **完善对角线分组逻辑**
   - 确保按 r+c = const 正确分组
   - 处理边界情况（对角线长度变化）

2. **实现对角线内部排序**
   - 每条对角线按反对角线方向排序
   - 从左上到右下，或从右下到左上

3. **实现之字形方向**
   - 偶数对角线：从左上到右下（正向）
   - 奇数对角线：从右下到左上（反向，需要反转）

**验证方法：**
- 可视化扫描路径（生成 28×28 索引图）
- 与 JPEG zigzag 扫描对比（确保逻辑一致）
- 测试对角线扫描的压缩效果
- 单元测试确保覆盖所有像素且无重复

**参考实现：**
- JPEG Zigzag 扫描算法
- 8×8 DCT 块的 zigzag 顺序

#### 2. 按策略分组可视化 ⚠️ **待实现**

**当前状态：**
- ✅ 已有聚合训练曲线（所有方法叠加在一张图上）
- ⚠️ **待实现**：按策略分组绘制（每个策略一张图）

**实现计划：**
- 修改 `visualize_results.py` 中的 `plot_training_curves` 函数
- 按策略分组（row/col/snake/diag）
- 每种策略内的方法用不同颜色线条
- 生成多组对比图

**可视化方案：**
```
results/
├── training_curves_all.png              # 当前：所有方法叠加
└── training_curves_by_strategy/
    ├── row_strategy.png                 # row策略：MLP, Transformer-BERT, BPE+Transformer-BERT
    ├── col_strategy.png                 # col策略：同上
    ├── snake_strategy.png               # snake策略：同上
    └── diag_strategy.png                # diag策略：同上（实现diag后）
```

**实现要点：**
- 从模型名称提取策略信息（如 `mlp_row` → `row`）
- 按策略分组结果
- 每个策略内，不同方法用不同颜色
- 图例标注清楚（MLP/Transformer/BPE+Transformer）

#### 3. 完整实验运行 📋 **实验计划**

**实验矩阵：**
```
策略数量：4种（row, col, snake, diag）
模型数量：
  - MLP × 4策略 = 4个实验
  - Transformer-BERT × 4策略 = 4个实验  
  - BPE+Transformer-BERT × 4策略 = 4个实验（每个策略需要先训练BPE）
  - LeNet-5 × 1 = 1个实验（不依赖策略）
总计：13个实验（diag实现前为12个）
```

**运行方式：**
```bash
# 自动运行所有策略的实验（待创建脚本）
./run_all_strategies.sh

# 或手动运行
# 1. 训练各策略的BPE模型
for strategy in row col snake diag; do
    python train_bpe.py --flatten_strategy $strategy --num_merges 200 --min_frequency 100
done

# 2. 训练所有模型
# ...（详细步骤见使用指南）
```

**实验时间估算：**
- BPE 训练：~1s × 4 = 4s（CPU）
- MLP 训练：~10s × 4 = 40s（GPU）
- Transformer 训练：~120s × 4 = 480s（GPU，8分钟）
- BPE+Transformer 训练：~120s × 4 = 480s（GPU，8分钟）
- LeNet-5：~60s（GPU）
- **总计：约 20 分钟**（20 epochs）

### 中期计划（1-2月）

#### 1. 扩展到 CIFAR-10

**挑战：**
- 图像尺寸：32×32×3 = 3072 长度（相比 MNIST 的 784 更长）
- 三通道：需要处理 RGB
- 更复杂的分类任务（10 类，但更复杂）

**序列化策略调整：**
- 考虑按通道分别展平 vs 交替展平
- 对角线扫描在多通道上的应用

#### 2. 策略组合实验

**新想法：**
- 混合策略：局部用 snake，全局用 diag
- 自适应策略：根据图像内容选择策略
- 多策略集成：用多个策略训练，然后 ensemble

#### 3. BPE 模式分析

**深度分析：**
- BPE 学到了什么模式？
- 不同策略下，BPE 发现的模式是否不同？
- 可视化 BPE 合并的像素对

### 长期计划（3-6月）

#### 1. 扩展到更大图像

**目标数据集：**
- ImageNet-32（32×32）
- Tiny ImageNet（64×64）

**技术挑战：**
- 序列长度大幅增加（1024, 4096）
- 需要更长的 Transformer 上下文窗口
- BPE 压缩率需要优化

#### 2. 2D BPE

**创新点：**
- 不仅合并相邻像素，还合并 2D 局部块
- 类似卷积的感受野，但用 BPE 发现
- 可能发现更复杂的空间模式

#### 3. 与其他方法对比

**对比方法：**
- Vision Transformer (ViT)
- Swin Transformer
- 其他图像序列化方法

**评估维度：**
- 准确率
- 训练效率
- 可解释性
- 迁移学习能力

---

## 技术细节

### BPE 实现细节

#### C++ 后端优势

**性能对比：**
- Python 后端：训练 ~40s（60k 样本，200 merges）
- C++ 后端：训练 ~1s（**40 倍加速**）

**技术栈：**
- C++ 实现核心 BPE 算法
- pybind11 绑定 Python 接口
- NumPy 数组传递（零拷贝）

**使用方式：**
```python
from bpe.bpe_engine import BPEEngine

# 创建引擎（默认 C++ 后端）
engine = BPEEngine(
    train_backend="cpp",   # C++ 后端
    encode_backend="cpp"   # C++ 后端
)

# 训练
stats = engine.train(
    token_sequences=sequences,
    num_merges=200,
    min_frequency=100
)

# 编码
engine.build_encoder()
encoded = engine.batch_encode(sequences)
```

#### BPE 压缩效果

**典型压缩率：**
- 原始序列：784 长度（MNIST）
- BPE 压缩后：~140 长度
- **压缩率：~18%**

**影响因素：**
- `num_merges`: 合并次数（默认 200）
- `min_frequency`: 最小频率阈值（默认 100）
- 序列化策略：不同策略可能影响压缩率

### Transformer 实现细节

#### 架构配置

**BERT-Small：**
```python
{
    'vocab_size': 256 + 8,        # 灰度值(256) + 特殊token(8)
    'd_model': 256,               # 隐藏维度
    'n_layers': 4,                 # Transformer层数
    'n_heads': 4,                  # 注意力头数
    'd_ff': 1024,                  # 前馈网络维度
    'max_seq_length': 1024,        # 最大序列长度
    'dropout': 0.1,
    'reset_weights': True          # 从头训练
}
```

#### 特殊 Token 处理

**Token ID 分配：**
- 灰度值 token: 0-255
- 特殊 token: 256-263（从 vocab_size 开始，避免冲突）
  - PAD: 256
  - UNK: 257
  - MASK: 258
  - CLS: 259
  - SEP: 260
  - ...

**设计考虑：**
- 避免与灰度值冲突
- 便于扩展（BPE 后的 vocab_size 更大）

### 训练配置

#### 超参数

**MLP：**
- 学习率：1e-3
- 优化器：Adam (betas=(0.9, 0.999))
- 权重衰减：1e-5

**LeNet-5：**
- 学习率：1e-3
- 优化器：Adam

**Transformer：**
- 学习率：5e-5（较小，因为从头训练）
- 优化器：Adam
- 学习率调度：CosineAnnealing（warmup 2 epochs）

#### 训练设置

- **Batch Size**: 128
- **Epochs**: 20（默认）
- **Validation Ratio**: 0.1（从训练集划分）
- **设备**: CUDA（强制，无 GPU 将退出）

---

## 使用指南

### 环境设置

#### 1. 依赖安装

```bash
# 基础依赖
pip install torch torchvision numpy tqdm matplotlib

# BPE C++ 后端
pip install pybind11
cd bpe
python setup.py build_ext --inplace
cd ..
```

#### 2. 验证环境

```bash
# 测试 BPE C++ 后端
python -c "from bpe.bpe_engine import BPEEngine; print('BPE C++ backend OK')"

# 测试数据加载
python -c "from bpe_data import get_mnist_dataloaders; print('Data loader OK')"
```

### 运行实验

#### 1. 单个实验

```bash
# MLP (row策略)
python train_mlp.py --flatten_strategy row --epochs 20

# Transformer (snake策略)
python train_transformer.py --transformer_type bert --flatten_strategy snake --epochs 20

# BPE+Transformer (diag策略，待实现)
python train_bpe_transformer.py --transformer_type bert --flatten_strategy diag --epochs 20
```

#### 2. 完整实验流程

**步骤 1：训练 BPE 模型（按策略）**
```bash
# row策略的BPE
python train_bpe.py --flatten_strategy row --num_merges 200 --min_frequency 100

# snake策略的BPE
python train_bpe.py --flatten_strategy snake --num_merges 200 --min_frequency 100
```

**步骤 2：训练基线模型**
```bash
# MLP (所有策略)
for strategy in row col snake; do
    python train_mlp.py --flatten_strategy $strategy --epochs 20
done

# LeNet-5 (不依赖策略)
python train_lenet.py --epochs 20
```

**步骤 3：训练 Transformer**
```bash
# Transformer-BERT (所有策略)
for strategy in row col snake; do
    python train_transformer.py --transformer_type bert --flatten_strategy $strategy --epochs 20
done
```

**步骤 4：训练 BPE+Transformer**
```bash
# BPE+Transformer-BERT (所有策略，需要先训练BPE)
for strategy in row col snake; do
    python train_bpe_transformer.py --transformer_type bert --flatten_strategy $strategy --epochs 20
done
```

#### 3. 评估与可视化

```bash
# 评估所有模型
python evaluate.py

# 生成对比报告
python compare_results.py

# 生成可视化图表
python visualize_results.py
```

### 结果文件

#### 结果目录结构

```
results/
├── mlp_row_results.json
├── mlp_col_results.json
├── mlp_snake_results.json
├── transformer_bert_row_results.json
├── transformer_bert_col_results.json
├── transformer_bert_snake_results.json
├── bpe_transformer_bert_row_results.json
├── bpe_transformer_bert_col_results.json
├── bpe_transformer_bert_snake_results.json
├── model_comparison.csv
├── training_curves_all.png
├── accuracy_comparison.png
└── efficiency_comparison.png
```

#### BPE 模型目录

```
saved_bpe_models/
├── mnist_bpe_row.pkl
├── mnist_bpe_col.pkl
├── mnist_bpe_snake.pkl
└── mnist_bpe_diag.pkl
```

---

## 实验预期

### 预期性能

**基线参考：**
- LeNet-5: ~99% 测试准确率（性能上限）
- MLP: ~97-98% 测试准确率（简单但有效）

**Transformer 预期：**
- Transformer (row): ~95-97%
- Transformer (snake): 可能略好于 row（减少跳变）
- BPE+Transformer: 可能略好或略差（取决于压缩质量）

**策略影响预期：**
- snake 和 diag 可能比 row/col 略好（更好的空间连续性）
- 不同策略对 BPE 压缩率可能有显著影响

### 研究价值

**如果实验成功，将证明：**
1. BPE 可以在图像数据上发现有意义模式
2. 序列化策略对 Transformer 建模有重要影响
3. 简单的序列化+BPE+Transformer 可以接近 CNN 性能

**如果实验失败，将揭示：**
1. Transformer 在展平序列上的局限性
2. BPE 在图像数据上的不足
3. 需要更复杂的序列化策略或架构改进

**无论成功与否，都有研究价值：**
- 系统性的对比实验
- 可重现的实验设计
- 为后续研究提供基础

---

## 关键技术决策

### 为什么使用 C++ BPE 后端？

**性能考虑：**
- BPE 训练是 CPU 密集型任务
- Python 实现太慢（~40s vs ~1s）
- C++ 后端提升 40 倍，不影响训练流程

**工程考虑：**
- 已编译的扩展模块，无需运行时编译
- 如果 C++ 后端不可用，可以回退到 Python（但性能较差）

### 为什么强制使用 CUDA？

**训练效率：**
- Transformer 训练需要 GPU 加速
- CPU 训练太慢（每个 epoch ~10 分钟 vs ~1 分钟）
- 确保实验可重现性和一致性

**工程决策：**
- 如果无 GPU，脚本直接退出并提示
- 避免用户在不合适的环境下浪费时间

### 为什么模型名称包含策略？

**实验组织：**
- 便于对比不同策略的结果
- 防止实验结果混淆
- 支持批量实验和自动化脚本

**命名规范：**
- `mlp_{strategy}` - MLP 模型
- `transformer_{type}_{strategy}` - Transformer 模型
- `bpe_transformer_{type}_{strategy}` - BPE+Transformer 模型

---

## 开发注意事项

### 代码风格

1. **模块化设计**
   - 每个模块职责单一
   - 模块间通过清晰接口交互

2. **配置驱动**
   - 所有超参数在 `bpe_config.py` 中统一管理
   - 支持命令行参数覆盖

3. **日志规范**
   - 使用统一的 logger
   - 详细记录训练过程
   - 便于问题排查

### 实验可重现性

1. **固定随机种子**
   - 所有随机操作使用固定种子
   - 确保实验可重现

2. **完整记录**
   - 每个实验保存完整配置
   - 记录所有超参数和模型结构

3. **版本控制**
   - 代码变更通过 git 管理
   - 重要实验打 tag

### 性能优化

1. **BPE 使用 C++ 后端**
   - 训练和编码都使用 C++
   - 批量处理优化

2. **数据加载优化**
   - 使用多进程 DataLoader
   - 预加载数据到内存（小数据集）

3. **模型优化**
   - 使用混合精度训练（可选）
   - 梯度累积（如果需要更大的 batch size）

---

## 总结

本项目是一个探索性研究项目，旨在验证 **BPE+Transformer** 在图像数据上的有效性。通过系统性的对比实验，我们希望回答：

1. **BPE 在图像上是否有效？**
2. **序列化策略是否重要？**
3. **简单方法能否接近传统方法？**

项目设计遵循：
- **独立性** - 可独立部署和运行
- **模块化** - 清晰的结构和接口
- **可扩展** - 易于添加新策略和新方法
- **科研友好** - 完整的记录和可视化

**下一步重点：**
1. 完善对角线扫描策略（diag）
2. 实现按策略分组的可视化
3. 运行完整实验，收集结果
4. 分析不同策略的效果差异

---

**文档维护：**
- 本文档应在项目迁移后继续更新
- 重要决策和技术细节应在此记录
- 实验发现和结论应及时补充

**最后更新：** 2025-10-30

