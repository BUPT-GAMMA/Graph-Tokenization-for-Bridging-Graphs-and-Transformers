# BPE for Images - 图像BPE压缩实验

## 项目概述

这是一个探索BPE（Byte Pair Encoding）在图像数据上应用效果的实验子项目。我们将MNIST图像展平为灰度值序列，然后对比以下几种方法：

1. **MLP Baseline** - 简单的多层感知机
2. **LeNet-5 CNN Baseline** - 经典卷积神经网络
3. **Transformer（灰度值直接作为token）** - 使用BERT/GTE架构
4. **BPE+Transformer** - BPE压缩后再使用Transformer

## 项目结构

```
bpe_for_images/
├── config.py                        # 配置文件
├── data/                            # 数据模块
│   ├── mnist_loader.py             # MNIST数据加载器
│   └── bpe_processor.py            # BPE处理器
├── models/                          # 模型模块
│   ├── mlp_classifier.py           # MLP分类器
│   ├── lenet.py                    # LeNet-5
│   └── transformer_classifier.py   # Transformer分类器
├── training_utils.py                # 训练工具函数
├── train_mlp.py                     # 训练MLP
├── train_lenet.py                   # 训练LeNet
├── train_transformer.py             # 训练Transformer
├── train_bpe.py                     # 训练BPE模型
├── train_bpe_transformer.py         # 训练BPE+Transformer
├── evaluate.py                      # 统一评估
├── compare_results.py               # 结果对比
├── visualize_results.py             # 结果可视化
├── results/                         # 实验结果
├── checkpoints/                     # 模型检查点
└── README.md                        # 本文档
```

## 实验设计

### 数据格式

- **原始数据**: MNIST手写数字 (28×28灰度图像)
- **展平序列**: 784长度的灰度值序列 (0-255)
- **BPE压缩**: 通过合并操作将序列压缩到约600左右

### 模型配置

#### 1. MLP
```python
输入: 784维向量
架构: 784 -> 512 -> 256 -> 10
激活: ReLU
Dropout: 0.2
```

#### 2. LeNet-5
```python
输入: 1×28×28图像
架构: Conv(6)→Pool→Conv(16)→Pool→FC(120)→FC(84)→FC(10)
```

#### 3. Transformer (BERT-small)
```python
输入: 灰度值序列 (vocab_size=256, seq_len=784)
架构:
  - d_model: 256
  - n_layers: 4
  - n_heads: 4
  - d_ff: 1024
Pooling: CLS token
分类头: 256 -> 128 -> 10
```

#### 4. BPE配置
```python
合并次数: 200
最小频率: 100
期望压缩率: ~75%
```

## 快速开始

### 环境要求

```bash
# 需要主项目的依赖
cd /home/gzy/py/tokenizerGraph
pip install -r requirements.txt

# 额外需要matplotlib用于可视化
pip install matplotlib
```

### 运行完整实验流程

```bash
cd /home/gzy/py/tokenizerGraph/bpe_for_images

# 1. 训练Baseline模型
python train_mlp.py
python train_lenet.py

# 2. 训练Transformer（灰度值直接作为token）
python train_transformer.py --transformer_type bert
python train_transformer.py --transformer_type gte

# 3. 训练BPE模型
python train_bpe.py

# 4. 训练BPE+Transformer
python train_bpe_transformer.py --transformer_type bert
python train_bpe_transformer.py --transformer_type gte

# 5. 评估所有模型
python evaluate.py

# 6. 生成对比报告
python compare_results.py

# 7. 可视化结果
python visualize_results.py
```

### 单独运行某个实验

```bash
# 训练MLP（自定义参数）
python train_mlp.py --epochs 30 --batch_size 256 --lr 0.001

# 训练LeNet
python train_lenet.py --epochs 20

# 训练Transformer（BERT）
python train_transformer.py --transformer_type bert --epochs 20 --lr 5e-5

# 训练BPE模型（自定义压缩参数）
python train_bpe.py --num_merges 300 --min_frequency 50

# 训练BPE+Transformer
python train_bpe_transformer.py --transformer_type bert --bpe_model ./bpe_models/mnist_bpe.pkl
```

## 配置说明

主要配置项位于 `config.py`：

```python
# 数据配置
BATCH_SIZE = 128
EPOCHS = 20
VAL_RATIO = 0.1

# 学习率
LR_MLP = 1e-3
LR_CNN = 1e-3
LR_TRANSFORMER = 5e-5

# BPE配置
BPE_NUM_MERGES = 200
BPE_MIN_FREQUENCY = 100

# Transformer配置
BERT_CONFIG = {
    'd_model': 256,
    'n_layers': 4,
    'n_heads': 4,
    ...
}
```

## 结果输出

### 训练结果 (results/)

每个模型训练后会生成JSON结果文件，包含：

```json
{
  "model_name": "mlp",
  "config": {...},
  "training_history": {
    "epochs": [1, 2, ...],
    "train_losses": [...],
    "train_accs": [...],
    "val_accs": [...],
    "epoch_times": [...]
  },
  "best_val_acc": 0.98,
  "final_test_acc": 0.97,
  "total_params": 123456,
  "training_time_total": 120.5
}
```

### 对比报告 (results/model_comparison.csv)

| model_name | test_acc | best_val_acc | total_params | training_time |
|------------|----------|--------------|--------------|---------------|
| lenet | 0.9900 | 0.9912 | 61,706 | 45.32 |
| mlp | 0.9850 | 0.9876 | 535,818 | 28.15 |
| ... | ... | ... | ... | ... |

### 可视化图表 (results/)

- `training_curves_all.png` - 所有模型的训练曲线汇总
- `accuracy_comparison.png` - 准确率对比柱状图
- `efficiency_comparison.png` - 参数量/训练时间 vs 准确率散点图
- `individual_curves/` - 每个模型的详细训练曲线

## 实验目标

### 主要研究问题

1. **BPE是否有效？** 
   - 对比 Transformer vs BPE+Transformer 的性能
   - 分析序列压缩带来的影响

2. **Transformer vs CNN/MLP？**
   - 在简单图像任务上的性能对比
   - 训练效率和参数量对比

3. **序列长度的影响？**
   - 784长度序列对Transformer的挑战
   - BPE压缩后的性能提升

### 预期结果

- LeNet-5应该达到最佳性能（~99%）
- MLP应该有良好的baseline性能（~98%）
- Transformer可能受限于序列长度和数据量
- BPE+Transformer可能通过压缩序列获得改善

## 扩展性

### 未来实验方向

1. **CIFAR-10**
   - 更大的图像尺寸 (32×32×3 = 3072长度)
   - 更复杂的分类任务
   - BPE压缩的必要性更强

2. **不同BPE策略**
   - 调整合并次数和频率阈值
   - 2D BPE（考虑空间结构）
   - 分通道BPE vs 联合BPE

3. **其他Transformer架构**
   - ViT (Vision Transformer)
   - Swin Transformer
   - 与BPE结合的可能性

### 代码扩展

项目设计考虑了扩展性：

- **数据加载器**: 易于添加新数据集（修改 `data/mnist_loader.py`）
- **模型**: 继承base类添加新模型（`models/`）
- **BPE**: 通用的BPE处理器（复用主项目实现）
- **训练流程**: 统一的训练工具（`training_utils.py`）

## 复用主项目代码

本项目复用了主项目的以下组件：

1. **BPE实现** - `src/algorithms/compression/main_bpe.py`
2. **Transformer编码器** - `src/models/unified_encoder.py`
3. **BERT模型** - `src/models/bert/`
4. **工具函数** - `src/utils/logger.py` 等

这确保了代码质量和与主项目的一致性。

## 注意事项

1. **设备选择**: 默认使用CUDA，可通过 `--device cpu` 切换
2. **内存使用**: Transformer模型需要较大内存（~4GB GPU）
3. **训练时间**: 
   - MLP/LeNet: ~30秒/epoch
   - Transformer: ~120秒/epoch（取决于硬件）
4. **BPE训练**: 需要先运行 `train_bpe.py` 才能运行BPE+Transformer

## 引用

如果使用本实验设计，请引用：

- LeNet: LeCun et al., 1998
- BPE: Sennrich et al., 2016
- BERT: Devlin et al., 2019

## 联系

如有问题或建议，请提issue或联系项目维护者。

## License

本项目继承主项目的License。

