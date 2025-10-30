"""
BPE for Images 实验配置
========================

独立的子项目配置，用于MNIST图像分类实验
"""

import os
from pathlib import Path

# ============== 路径配置 ==============
PROJECT_ROOT = Path(__file__).parent
MAIN_PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "mnist_data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
BPE_MODEL_DIR = PROJECT_ROOT / "bpe_models"

# 确保目录存在
for dir_path in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR, BPE_MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============== 数据配置 ==============
# MNIST配置
IMAGE_SIZE = 28
FLATTEN_SIZE = IMAGE_SIZE * IMAGE_SIZE  # 784
NUM_CLASSES = 10
GRAYSCALE_VOCAB_SIZE = 256  # 0-255灰度值

# 数据集划分
TRAIN_SIZE = 60000
TEST_SIZE = 10000
VAL_RATIO = 0.1  # 从训练集中划分验证集


# ============== BPE配置 ==============
BPE_NUM_MERGES = 200          # BPE合并次数，压缩784->~600
BPE_MIN_FREQUENCY = 100       # 最小频率阈值（60k样本规模适用）
BPE_BACKEND = "python"        # 使用Python后端（简单场景）


# ============== 训练配置 ==============
# 通用训练参数
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 20
DEVICE = "cuda"  # "cuda" or "cpu"
SEED = 42

# 学习率（不同模型）
LR_MLP = 1e-3
LR_CNN = 1e-3
LR_TRANSFORMER = 5e-5

# 优化器配置
WEIGHT_DECAY = 1e-5
ADAM_BETAS = (0.9, 0.999)

# 学习率调度
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"  # "cosine" or "step"
LR_WARMUP_EPOCHS = 2


# ============== 模型配置 ==============

# MLP配置
MLP_CONFIG = {
    "input_size": FLATTEN_SIZE,  # 784
    "hidden_sizes": [512, 256],
    "num_classes": NUM_CLASSES,
    "dropout": 0.2,
    "activation": "relu"
}

# LeNet-5配置
LENET_CONFIG = {
    "in_channels": 1,
    "num_classes": NUM_CLASSES,
    "conv1_out": 6,
    "conv2_out": 16,
    "fc1_out": 120,
    "fc2_out": 84
}

# Transformer配置（BERT-small）
BERT_CONFIG = {
    "vocab_size": GRAYSCALE_VOCAB_SIZE,  # 初始256，BPE后会增加
    "d_model": 256,           # hidden_size，相比主项目的512减小
    "n_heads": 4,             # attention heads
    "n_layers": 4,            # transformer layers
    "d_ff": 1024,             # feedforward dimension (4 * d_model)
    "max_seq_length": 1024,   # 足够长以容纳展平序列
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation": "gelu",
    "layer_norm_eps": 1e-12,
    "initializer_range": 0.02,
    "pooling_method": "cls"   # 使用CLS token pooling
}

# GTE配置
GTE_CONFIG = {
    "model_path": str(MAIN_PROJECT_ROOT / "gte_model"),
    "use_pretrained": False,   # 随机初始化，不使用预训练权重
    "vocab_size": GRAYSCALE_VOCAB_SIZE,
    "max_seq_length": 1024,
    "pooling_method": "mean"   # GTE通常使用mean pooling
}

# Transformer类型选择
TRANSFORMER_TYPES = ["bert", "gte"]


# ============== 日志和保存配置 ==============
LOG_INTERVAL = 50  # 每多少个batch打印一次
SAVE_EVERY_EPOCH = True
SAVE_BEST_ONLY = True  # 只保存最佳模型
EVAL_EVERY_EPOCH = True


# ============== 实验配置 ==============
# 要运行的实验
EXPERIMENTS = [
    "mlp",
    "lenet",
    "transformer_bert",
    "transformer_gte",
    "bpe_transformer_bert",
    "bpe_transformer_gte"
]

# 实验结果文件名
RESULTS_FILE = "experiment_results.json"
COMPARISON_CSV = "model_comparison.csv"
TRAINING_CURVES_FILE = "training_curves.png"


# ============== 工具函数 ==============
def get_checkpoint_path(model_name: str, epoch: int = None) -> Path:
    """获取模型检查点路径"""
    if epoch is not None:
        return CHECKPOINTS_DIR / f"{model_name}_epoch{epoch}.pth"
    return CHECKPOINTS_DIR / f"{model_name}_best.pth"


def get_result_path(model_name: str) -> Path:
    """获取结果JSON文件路径"""
    return RESULTS_DIR / f"{model_name}_results.json"


def get_bpe_model_path() -> Path:
    """获取BPE模型保存路径"""
    return BPE_MODEL_DIR / "mnist_bpe.pkl"


# ============== 显示配置摘要 ==============
def print_config_summary():
    """打印配置摘要"""
    print("\n" + "="*60)
    print("BPE for Images - 实验配置")
    print("="*60)
    print(f"数据集: MNIST ({TRAIN_SIZE} train, {TEST_SIZE} test)")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE} -> {FLATTEN_SIZE} flatten")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"BPE配置: {BPE_NUM_MERGES} merges, min_freq={BPE_MIN_FREQUENCY}")
    print(f"Transformer: BERT({BERT_CONFIG['d_model']}d/{BERT_CONFIG['n_layers']}L)")
    print(f"设备: {DEVICE}")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_config_summary()

