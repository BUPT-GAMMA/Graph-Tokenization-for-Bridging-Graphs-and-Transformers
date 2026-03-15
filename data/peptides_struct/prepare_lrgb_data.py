#!/usr/bin/env python3
"""
LRGB数据集预处理脚本
===================

从PyTorch Geometric下载LRGB数据集并转换为项目格式。
"""

import os
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Any, Dict
import gzip

try:
    import torch_geometric
    from torch_geometric.datasets import LRGBDataset
    from torch_geometric.data import Data
    from torch_geometric.utils import to_dgl
except ImportError:
    print("请安装torch_geometric: pip install torch_geometric")
    exit(1)

import dgl
from config import ProjectConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def scan_and_build_token_mappings(train_dataset, val_dataset, test_dataset):
    """扫描所有数据集，构建特征组合到token的映射"""
    logger.info("🔍 扫描所有数据，发现特征组合...")
    
    node_combinations = set()
    edge_combinations = set()
    
    # 扫描所有数据集
    for dataset_name, dataset in [("train", train_dataset), ("val", val_dataset), ("test", test_dataset)]:
        logger.info(f"📊 扫描 {dataset_name} 数据: {len(dataset)} 个图")
        
        for i, pyg_data in enumerate(dataset):
            if i % 1000 == 0 and i > 0:
                logger.info(f"  扫描进度: {i}/{len(dataset)}")
            
            try:
                g = convert_to_dgl(pyg_data)
            except Exception:
                continue  # 跳过转换失败的图
            
            # 收集节点特征组合
            node_features = g.ndata['x']
            for j in range(node_features.size(0)):
                node_combo = tuple(node_features[j].tolist())
                node_combinations.add(node_combo)
            
            # 收集边特征组合
            edge_features = g.edata['edge_attr']
            for j in range(edge_features.size(0)):
                edge_combo = tuple(edge_features[j].tolist())
                edge_combinations.add(edge_combo)
    
    # 转换为列表并排序（确保确定性）
    node_combinations = sorted(list(node_combinations))
    edge_combinations = sorted(list(edge_combinations))
    
    logger.info(f"🎯 发现特征组合:")
    logger.info(f"  节点特征组合: {len(node_combinations)} 种")
    logger.info(f"  边特征组合: {len(edge_combinations)} 种")
    
    # 创建映射字典
    node_combo_to_token = {}
    edge_combo_to_token = {}
    
    # 节点特征组合 -> 奇数域token（从1开始）
    for i, combo in enumerate(node_combinations):
        node_combo_to_token[combo] = 2 * i + 1
    
    # 边特征组合 -> 偶数域token（从0开始）
    for i, combo in enumerate(edge_combinations):
        edge_combo_to_token[combo] = 2 * i
    
    logger.info(f"🔧 Token映射创建完成:")
    logger.info(f"  节点tokens: 奇数域 [1, 3, 5, ..., {2 * len(node_combinations) - 1}]")
    logger.info(f"  边tokens: 偶数域 [0, 2, 4, ..., {2 * len(edge_combinations) - 2}]")
    
    return node_combo_to_token, edge_combo_to_token


def compute_tokens_for_graph(dgl_graph, node_combo_to_token, edge_combo_to_token):
    """为DGL图计算node_token_ids和edge_token_ids"""
    # 计算节点tokens
    node_features = dgl_graph.ndata['x']
    node_tokens = []
    node_type_ids = []
    
    for i in range(node_features.size(0)):
        node_feat_tuple = tuple(node_features[i].tolist())
        token = node_combo_to_token.get(node_feat_tuple, 1)  # 默认token为1（未知节点类型）
        node_tokens.append(token)
        node_type_ids.append((token - 1) // 2)  # 从token反推type_id
    
    dgl_graph.ndata['node_token_ids'] = torch.tensor(node_tokens, dtype=torch.long).view(-1, 1)
    dgl_graph.ndata['node_type_id'] = torch.tensor(node_type_ids, dtype=torch.long)
    
    # 计算边tokens
    edge_features = dgl_graph.edata['edge_attr']
    edge_tokens = []
    edge_type_ids = []
    
    for i in range(edge_features.size(0)):
        edge_feat_tuple = tuple(edge_features[i].tolist())
        token = edge_combo_to_token.get(edge_feat_tuple, 0)  # 默认token为0（未知边类型）
        edge_tokens.append(token)
        edge_type_ids.append(token // 2)  # 从token反推type_id
    
    dgl_graph.edata['edge_token_ids'] = torch.tensor(edge_tokens, dtype=torch.long).view(-1, 1)
    dgl_graph.edata['edge_type_id'] = torch.tensor(edge_type_ids, dtype=torch.long)


def extract_lightweight_graph_data(dgl_graph) -> Dict[str, Any]:
    """
    从DGL图中提取轻量级数据用于存储（不包含标签）
    
    Args:
        dgl_graph: DGL图对象
        
    Returns:
        包含最小化图数据的字典
    """
    # 获取图结构
    src, dst = dgl_graph.edges()
    
    graph_data = {
        'num_nodes': int(dgl_graph.num_nodes()),
        'num_edges': int(dgl_graph.num_edges()),
        'edges': (src.numpy().astype(np.int32), dst.numpy().astype(np.int32)),  # 边列表，numpy格式更紧凑
    }
    
    # 节点特征：PyG->DGL转换后存储在'x'中
    graph_data['node_features'] = dgl_graph.ndata['x'].numpy().astype(np.float32)
    
    # 边特征：PyG->DGL转换后存储在'edge_attr'中  
    graph_data['edge_features'] = dgl_graph.edata['edge_attr'].numpy().astype(np.float32)
    
    # Token IDs（预计算的）
    graph_data['node_token_ids'] = dgl_graph.ndata['node_token_ids'].numpy().astype(np.int32)
    
    graph_data['edge_token_ids'] = dgl_graph.edata['edge_token_ids'].numpy().astype(np.int32)
    
    # 类型IDs（预计算的）
    graph_data['node_type_ids'] = dgl_graph.ndata['node_type_id'].numpy().astype(np.int32)
    
    graph_data['edge_type_ids'] = dgl_graph.edata['edge_type_id'].numpy().astype(np.int32)
    
    return graph_data


def convert_to_dgl(pyg_data: Data) -> dgl.DGLGraph:
    """将PyG数据转换为DGL图"""
    try:
        # 使用torch_geometric的to_dgl函数
        g = to_dgl(pyg_data)
        
        # 确保图的特征格式正确
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            g.ndata['x'] = pyg_data.x
        if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
            g.edata['edge_attr'] = pyg_data.edge_attr
            
        return g
    except Exception as e:
        # logger.warning(f"使用to_dgl转换失败，手动创建DGL图: {e}")
        
        # 手动创建DGL图
        edge_index = pyg_data.edge_index
        num_nodes = pyg_data.num_nodes
        
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        
        # 复制节点特征
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            g.ndata['x'] = pyg_data.x
            
        # 复制边特征
        if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
            g.edata['edge_attr'] = pyg_data.edge_attr
        
        return g


def prepare_lrgb_dataset(dataset_name: str, original_name: str, config: ProjectConfig):
    """准备LRGB数据集"""
    logger.info(f"🔧 准备LRGB数据集: {original_name} -> {dataset_name}")
    
    # 数据集目录
    data_dir = Path(config.data_dir) / dataset_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载数据集
    logger.info(f"📥 下载 {original_name} 数据集...")
    train_dataset = LRGBDataset(root=str(data_dir / "raw"), name=original_name, split='train')
    val_dataset = LRGBDataset(root=str(data_dir / "raw"), name=original_name, split='val') 
    test_dataset = LRGBDataset(root=str(data_dir / "raw"), name=original_name, split='test')
    
    logger.info(f"📊 数据集大小: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # 扫描并创建token映射表
    node_combo_to_token, edge_combo_to_token = scan_and_build_token_mappings(train_dataset, val_dataset, test_dataset)
    
    # 转换数据 - 直接保存轻量级格式
    all_data = []
    
    def process_split(dataset, split_name):
        """处理单个数据划分"""
        logger.info(f"🔄 处理 {split_name} 数据...")
        split_indices = []
        success_count = 0
        failure_count = 0
        
        for i, pyg_data in enumerate(dataset):
            # 转换为DGL图
            try:
                g = convert_to_dgl(pyg_data)
                success_count += 1
            except Exception as e:
                failure_count += 1
                if failure_count <= 5:  # 只记录前5个失败样本的详细信息
                    logger.warning(f"转换 {split_name} 第{i}个图失败 (节点:{pyg_data.num_nodes}, 边:{pyg_data.num_edges}): {e}")
                continue
            
            # 计算并添加token IDs
            compute_tokens_for_graph(g, node_combo_to_token, edge_combo_to_token)
            
            # 获取和处理标签
            # PyG LRGB数据集的y格式确定：shape=(1, num_labels)的torch.Tensor
            label_tensor = pyg_data.y  # 直接使用，不做假设性检查
            
            if original_name == "Peptides-func":
                # Peptides-func: shape=(1, 10)，转换为(10,)的numpy数组（多标签分类）
                processed_label = label_tensor.squeeze(0).numpy()  # (1,10) -> (10,)
            elif original_name == "Peptides-struct":
                # Peptides-struct: shape=(1, 11)，转换为(11,)的numpy数组（多目标回归）
                processed_label = label_tensor.squeeze(0).numpy()  # (1,11) -> (11,)
            else:
                raise ValueError(f"未知的LRGB数据集: {original_name}")
            
            # 转换为轻量级格式并添加到总数据列表
            lightweight_graph = extract_lightweight_graph_data(g)
            
            all_data.append((lightweight_graph, processed_label))
            split_indices.append(len(all_data) - 1)
        
        # 统计信息
        total_samples = len(dataset)
        success_rate = (success_count / total_samples) * 100 if total_samples > 0 else 0
        logger.info(f"📊 {split_name} 处理完成: ✅ {success_count}/{total_samples} ({success_rate:.2f}%), ❌ {failure_count} 失败")
        
        return split_indices, success_count, failure_count
    
    # 处理各个划分
    train_indices, train_success, train_failures = process_split(train_dataset, "train")
    val_indices, val_success, val_failures = process_split(val_dataset, "val")
    test_indices, test_success, test_failures = process_split(test_dataset, "test")
    
    # 总体统计
    total_success = train_success + val_success + test_success
    total_failures = train_failures + val_failures + test_failures
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    overall_success_rate = (total_success / total_samples) * 100 if total_samples > 0 else 0
    
    logger.info(f"🎯 数据集总体统计:")
    logger.info(f"   原始样本: {total_samples}")
    logger.info(f"   成功转换: {total_success} ({overall_success_rate:.2f}%)")
    logger.info(f"   转换失败: {total_failures}")
    logger.info(f"   最终数据: {len(all_data)}")
    
    # 保存数据
    logger.info("💾 保存处理后的轻量级数据...")
    
    # 保存主数据文件 - 使用压缩格式
    data_file = data_dir / "data.pkl.gz"
    with gzip.open(data_file, "wb") as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 统计文件大小
    file_size_mb = data_file.stat().st_size / (1024**2)
    avg_size_kb = (file_size_mb * 1024) / len(all_data) if all_data else 0
    logger.info(f"📦 轻量级数据文件: {file_size_mb:.1f} MB (平均 {avg_size_kb:.1f} KB/图)")
    
    # 保存索引文件
    with open(data_dir / "train_index.json", "w") as f:
        json.dump(train_indices, f)
    
    with open(data_dir / "val_index.json", "w") as f:
        json.dump(val_indices, f)
        
    with open(data_dir / "test_index.json", "w") as f:
        json.dump(test_indices, f)
    
    # 保存token映射信息
    token_mapping_file = data_dir / "token_mappings.json"
    token_mappings = {
        'node_combo_to_token': {str(k): v for k, v in node_combo_to_token.items()},
        'edge_combo_to_token': {str(k): v for k, v in edge_combo_to_token.items()},
        'num_node_combinations': len(node_combo_to_token),
        'num_edge_combinations': len(edge_combo_to_token),
        'max_node_token': max(node_combo_to_token.values()) if node_combo_to_token else 1,
        'max_edge_token': max(edge_combo_to_token.values()) if edge_combo_to_token else 0,
    }
    
    with open(token_mapping_file, "w") as f:
        json.dump(token_mappings, f, indent=2)
    
    logger.info(f"📋 Token映射已保存: {token_mapping_file}")
    logger.info(f"✅ {dataset_name} 数据集准备完成!")
    logger.info(f"📁 数据保存在: {data_dir}")
    
    return len(all_data), len(train_indices), len(val_indices), len(test_indices), total_failures


def main():
    """主函数"""
    logger.info("🚀 开始准备LRGB数据集...")
    
    # 加载配置
    config = ProjectConfig()
    
    datasets = ["Peptides-func", "Peptides-struct"]
    
    # 规范化数据集名称映射
    dataset_mapping = {
        "Peptides-func": "peptides_func",
        "Peptides-struct": "peptides_struct"
    }
    
    for original_name in datasets:
        normalized_name = dataset_mapping[original_name]
        
        logger.info(f"🔧 处理数据集: {original_name} -> {normalized_name}")
        
        try:
            total, train, val, test, failures = prepare_lrgb_dataset(normalized_name, original_name, config)
            
            logger.info(f"📋 {normalized_name} 最终统计:")
            logger.info(f"  有效数据: {total}")
            logger.info(f"  训练集: {train}")
            logger.info(f"  验证集: {val}")
            logger.info(f"  测试集: {test}")
            logger.info(f"  失败样本: {failures}")
            
        except Exception as e:
            logger.error(f"❌ 处理 {original_name} 时出错: {e}")
            continue
    
    logger.info("🎉 所有LRGB数据集准备完成!")


if __name__ == "__main__":
    main()
