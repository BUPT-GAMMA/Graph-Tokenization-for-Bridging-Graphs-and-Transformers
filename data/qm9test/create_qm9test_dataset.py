#!/usr/bin/env python3
"""
创建QM9Test数据集脚本
==================

从QM9数据集中提取10%的数据，创建用于测试的QM9Test数据集。
按照新的数据结构保存：统一的数据文件 + 索引文件。
"""

import os
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
from tqdm import tqdm
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import ProjectConfig
from src.data.qm9_loader import QM9Loader
from utils.logger import get_logger

logger = get_logger(__name__)


def create_qm9test_dataset(config: ProjectConfig, test_ratio: float = 0.1, random_state: int = 42) -> None:
    """
    创建QM9Test数据集
    
    Args:
        config: 项目配置
        test_ratio: 测试数据比例，默认0.1（10%）
        random_state: 随机种子
    """
    logger.info(f"🚀 开始创建QM9Test数据集 (比例: {test_ratio*100}%)")
    
    # 创建QM9Test数据目录
    qm9test_dir = Path("data/qm9test")
    qm9test_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载完整的QM9数据
    logger.info("📂 加载完整QM9数据...")
    qm9_loader = QM9Loader(config)
    all_data, split_indices = qm9_loader.get_all_data_with_indices()
    
    logger.info(f"✅ 成功加载 {len(all_data)} 个QM9样本")
    
    # 随机选择10%的数据
    np.random.seed(random_state)
    total_size = len(all_data)
    test_size = int(total_size * test_ratio)
    
    # 随机选择索引
    selected_indices = np.random.choice(total_size, test_size, replace=False)
    selected_indices = sorted(selected_indices.tolist())  # 按顺序排列
    
    logger.info(f"🎯 随机选择 {test_size} 个样本 (索引范围: {min(selected_indices)} - {max(selected_indices)})")
    
    # 提取选中的数据
    test_data = [all_data[i] for i in selected_indices]
    
    # 创建新的划分（8:1:1）
    logger.info("📊 创建新的数据划分 (8:1:1)...")
    train_size = int(test_size * 0.8)
    val_size = int(test_size * 0.1)
    
    # 使用随机划分，而不是截取
    from sklearn.model_selection import train_test_split
    
    # 首先划分出训练集和临时集
    train_indices, temp_indices = train_test_split(
        selected_indices, 
        train_size=train_size, 
        random_state=random_state, 
        shuffle=True
    )
    
    # 然后从临时集中划分出验证集和测试集
    test_val_ratio = 0.5  # 验证集和测试集各占临时集的一半
    val_indices, test_indices = train_test_split(
        temp_indices, 
        train_size=test_val_ratio, 
        random_state=random_state, 
        shuffle=True
    )
    
    # 按顺序排列
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)
    
    # 创建新的索引映射（在test_data中的位置）
    train_new_indices = [selected_indices.index(i) for i in train_indices]
    val_new_indices = [selected_indices.index(i) for i in val_indices]
    test_new_indices = [selected_indices.index(i) for i in test_indices]
    
    logger.info(f"📈 划分完成:")
    logger.info(f"   - 训练集: {len(train_new_indices)} 个样本")
    logger.info(f"   - 验证集: {len(val_new_indices)} 个样本")
    logger.info(f"   - 测试集: {len(test_new_indices)} 个样本")
    
    # 保存统一的数据文件
    logger.info("💾 保存统一数据文件...")
    data_file = qm9test_dir / "data.pkl"
    
    # 转换为与QM9数据集一致的格式 (graph, properties) tuple
    formatted_data = []
    for sample in test_data:
        graph = sample['dgl_graph']
        properties = sample['properties']
        formatted_data.append((graph, properties))
    
    with open(data_file, 'wb') as f:
        pickle.dump(formatted_data, f)
    
    # 保存索引文件
    logger.info("💾 保存索引文件...")
    train_index_file = qm9test_dir / "train_index.json"
    val_index_file = qm9test_dir / "val_index.json"
    test_index_file = qm9test_dir / "test_index.json"
    
    with open(train_index_file, 'w') as f:
        json.dump(train_new_indices, f)
    with open(val_index_file, 'w') as f:
        json.dump(val_new_indices, f)
    with open(test_index_file, 'w') as f:
        json.dump(test_new_indices, f)
    
    # 提取并保存SMILES文件
    logger.info("💾 保存SMILES文件...")
    smiles_1 = []
    smiles_2 = []
    smiles_3 = []
    smiles_4 = []
    
    for sample in tqdm(test_data, desc="提取SMILES"):
        smiles_1.append(sample.get('smiles_1', ''))
        smiles_2.append(sample.get('smiles_2', ''))
        smiles_3.append(sample.get('smiles_3', ''))
        smiles_4.append(sample.get('smiles_4', ''))
    
    # 保存SMILES文件
    smiles_1_file = qm9test_dir / "smiles_1_direct.txt"
    smiles_2_file = qm9test_dir / "smiles_2_explicit_h.txt"
    smiles_3_file = qm9test_dir / "smiles_3_addhs.txt"
    smiles_4_file = qm9test_dir / "smiles_4_addhs_explicit_h.txt"
    
    with open(smiles_1_file, 'w') as f:
        f.write('\n'.join(smiles_1))
    with open(smiles_2_file, 'w') as f:
        f.write('\n'.join(smiles_2))
    with open(smiles_3_file, 'w') as f:
        f.write('\n'.join(smiles_3))
    with open(smiles_4_file, 'w') as f:
        f.write('\n'.join(smiles_4))
    
    # 保存元数据
    metadata = {
        'dataset_name': 'qm9test',
        'source_dataset': 'qm9',
        'test_ratio': test_ratio,
        'total_samples': len(test_data),
        'train_samples': len(train_new_indices),
        'val_samples': len(val_new_indices),
        'test_samples': len(test_new_indices),
        'original_indices': selected_indices,  # 在原始QM9中的索引
        'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'random_state': random_state
    }
    
    metadata_file = qm9test_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✅ QM9Test数据集创建完成！")
    logger.info(f"📁 数据目录: {qm9test_dir}")
    logger.info(f"📊 数据集统计:")
    logger.info(f"   - 总样本数: {len(test_data)}")
    logger.info(f"   - 训练集: {len(train_new_indices)}")
    logger.info(f"   - 验证集: {len(val_new_indices)}")
    logger.info(f"   - 测试集: {len(test_new_indices)}")
    logger.info(f"   - 原始QM9索引范围: {min(selected_indices)} - {max(selected_indices)}")


def verify_qm9test_dataset(config: ProjectConfig) -> None:
    """
    验证QM9Test数据集
    
    Args:
        config: 项目配置
    """
    logger.info("🔍 验证QM9Test数据集...")
    
    qm9test_dir = Path("data/qm9test")
    
    # 检查文件是否存在
    required_files = [
        "data.pkl",
        "train_index.json",
        "val_index.json", 
        "test_index.json",
        "smiles_1_direct.txt",
        "smiles_2_explicit_h.txt",
        "smiles_3_addhs.txt",
        "smiles_4_addhs_explicit_h.txt",
        "metadata.json"
    ]
    
    for file_name in required_files:
        file_path = qm9test_dir / file_name
        if not file_path.exists():
            logger.error(f"❌ 缺少文件: {file_name}")
            return
        logger.info(f"✅ 文件存在: {file_name}")
    
    # 加载数据验证
    try:
        # 加载数据
        with open(qm9test_dir / "data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # 加载索引
        with open(qm9test_dir / "train_index.json", 'r') as f:
            train_indices = json.load(f)
        with open(qm9test_dir / "val_index.json", 'r') as f:
            val_indices = json.load(f)
        with open(qm9test_dir / "test_index.json", 'r') as f:
            test_indices = json.load(f)
        
        # 验证索引
        all_indices = set(train_indices) | set(val_indices) | set(test_indices)
        assert len(all_indices) == len(train_indices) + len(val_indices) + len(test_indices), "索引有重复"
        assert max(all_indices) < len(data), "索引超出范围"
        assert train_indices == sorted(train_indices), "训练集索引未按顺序排列"
        assert val_indices == sorted(val_indices), "验证集索引未按顺序排列"
        assert test_indices == sorted(test_indices), "测试集索引未按顺序排列"
        
        # 验证SMILES
        with open(qm9test_dir / "smiles_1_direct.txt", 'r') as f:
            smiles_1 = f.read().strip().split('\n')
        
        assert len(smiles_1) == len(data), "SMILES数量与数据不匹配"
        
        logger.info("✅ QM9Test数据集验证通过！")
        logger.info(f"📊 验证结果:")
        logger.info(f"   - 数据样本数: {len(data)}")
        logger.info(f"   - 训练集索引: {len(train_indices)}")
        logger.info(f"   - 验证集索引: {len(val_indices)}")
        logger.info(f"   - 测试集索引: {len(test_indices)}")
        logger.info(f"   - SMILES数量: {len(smiles_1)}")
        
    except Exception as e:
        logger.error(f"❌ 数据集验证失败: {e}")
        raise


def main():
    """主函数"""
    logger.info("🚀 QM9Test数据集创建工具")
    
    # 加载配置
    config = ProjectConfig()
    
    # 创建QM9Test数据集
    create_qm9test_dataset(config, test_ratio=0.1, random_state=42)
    
    # 验证数据集
    verify_qm9test_dataset(config)
    
    logger.info("🎉 QM9Test数据集创建和验证完成！")


if __name__ == "__main__":
    main() 
