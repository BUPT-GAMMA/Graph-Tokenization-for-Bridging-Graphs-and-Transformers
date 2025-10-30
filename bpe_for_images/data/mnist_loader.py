"""
MNIST数据加载器
==============

提供三种数据格式：
1. 原始图像格式（用于CNN）
2. 展平向量格式（用于MLP）
3. 序列格式（用于Transformer/BPE）
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, List, Optional
import sys
from pathlib import Path

# 添加主项目路径以导入logger
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MNISTDataset(Dataset):
    """
    MNIST数据集包装器，支持多种输出格式
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        format: str = "image"  # "image", "flatten", "sequence"
    ):
        """
        Args:
            root: 数据根目录
            train: 是否为训练集
            download: 是否下载数据
            format: 输出格式
                - "image": (1, 28, 28) 图像张量
                - "flatten": (784,) 展平向量
                - "sequence": (784,) 灰度值序列（int类型，0-255）
        """
        self.format = format
        
        # 根据format选择transform
        if format in ["image", "flatten"]:
            # 转换为tensor并归一化到[0,1]
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:  # sequence格式
            # 不归一化，保持0-255的整数
            transform = None
        
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
        logger.info(f"加载MNIST数据集: {'训练集' if train else '测试集'}, "
                   f"样本数={len(self.dataset)}, 格式={format}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.format == "sequence":
            # 直接从PIL图像获取numpy数组
            img, label = self.dataset.data[idx], int(self.dataset.targets[idx])
            # 转换为numpy数组并展平
            img_array = img.numpy() if isinstance(img, torch.Tensor) else np.array(img)
            sequence = img_array.flatten()  # (784,) uint8
            return torch.from_numpy(sequence.astype(np.int64)), label
        else:
            img, label = self.dataset[idx]
            if self.format == "flatten":
                img = img.view(-1)  # (1, 28, 28) -> (784,)
            return img, label


def get_mnist_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    format: str = "image",
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取MNIST数据加载器（训练/验证/测试）
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        val_ratio: 验证集比例（从训练集划分）
        format: 数据格式（"image", "flatten", "sequence"）
        seed: 随机种子
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 加载数据集
    train_dataset = MNISTDataset(data_dir, train=True, download=True, format=format)
    test_dataset = MNISTDataset(data_dir, train=False, download=True, format=format)
    
    # 划分训练/验证集
    train_size = len(train_dataset)
    val_size = int(train_size * val_ratio)
    train_size = train_size - val_size
    
    # 使用固定随机种子确保可重复性
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据集划分: 训练={train_size}, 验证={val_size}, 测试={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_mnist_raw_data(
    data_dir: str,
    train: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取原始MNIST数据（用于BPE训练）
    
    Args:
        data_dir: 数据目录
        train: 是否为训练集
    
    Returns:
        (images, labels): 
            images: (N, 28, 28) uint8数组
            labels: (N,) int数组
    """
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True
    )
    
    images = dataset.data.numpy()  # (N, 28, 28)
    labels = dataset.targets.numpy()  # (N,)
    
    logger.info(f"获取原始MNIST数据: {'训练集' if train else '测试集'}, "
               f"形状={images.shape}")
    
    return images, labels


def prepare_flattened_sequences(
    images: np.ndarray
) -> List[List[int]]:
    """
    将图像转换为展平的灰度值序列（用于BPE训练）
    
    Args:
        images: (N, 28, 28) 图像数组
    
    Returns:
        List[List[int]]: N个长度为784的灰度值序列
    """
    N = images.shape[0]
    sequences = []
    
    for i in range(N):
        # 展平为(784,)并转换为Python int列表
        seq = images[i].flatten().astype(np.int64).tolist()
        sequences.append(seq)
    
    logger.info(f"准备展平序列: {N}个样本, 每个长度={len(sequences[0])}")
    
    return sequences


# ============== 测试代码 ==============
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR, BATCH_SIZE, VAL_RATIO, NUM_WORKERS
    
    print("测试MNIST数据加载器...")
    
    # 测试三种格式
    for format in ["image", "flatten", "sequence"]:
        print(f"\n测试格式: {format}")
        train_loader, val_loader, test_loader = get_mnist_dataloaders(
            data_dir=str(DATA_DIR),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            val_ratio=VAL_RATIO,
            format=format
        )
        
        # 获取一个batch
        batch_data, batch_labels = next(iter(train_loader))
        print(f"  训练batch形状: {batch_data.shape}, 标签: {batch_labels.shape}")
        print(f"  数据类型: {batch_data.dtype}, 范围: [{batch_data.min()}, {batch_data.max()}]")
    
    # 测试原始数据加载
    print("\n测试原始数据加载...")
    images, labels = get_mnist_raw_data(str(DATA_DIR), train=True)
    print(f"  图像形状: {images.shape}, 标签形状: {labels.shape}")
    
    # 测试序列准备
    print("\n测试序列准备...")
    sequences = prepare_flattened_sequences(images[:10])
    print(f"  序列数量: {len(sequences)}")
    print(f"  第一个序列长度: {len(sequences[0])}")
    print(f"  第一个序列样例（前10个值）: {sequences[0][:10]}")

