"""
QM9数据集处理脚本

使用现有的QM9数据加载器接口，直接从DGL图重建分子，
生成四种格式的SMILES，并按照train/val/test划分保存。
如果分子重建失败，使用原始SMILES保存四遍。
"""

import os
import json
import pickle
import torch
import dgl
from rdkit import Chem
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

# 导入项目模块
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.qm9_loader import QM9Loader
from config import ProjectConfig

def dgl_graph_to_mol(g: dgl.DGLGraph) -> Optional[Chem.Mol]:
    """
    从DGL图重建RDKit分子对象（忽略氢原子）
    
    Args:
        g: DGL图对象
    
    Returns:
        RDKit分子对象，转换失败返回None
    """
    try:
        # 获取节点和边信息
        u, v = g.edges()
        
        # 提取原子信息
        if 'attr' in g.ndata:
            # QM9数据集使用'attr'作为节点特征，第5维是原子序数
            node_features = g.ndata['attr']
            atomic_nums = [int(node_features[i][5].item()) for i in range(g.num_nodes())]
        elif 'feat' in g.ndata:
            # 备用方法：使用'feat'特征
            node_features = g.ndata['feat']
            atomic_nums = [int(node_features[i][5].item()) for i in range(g.num_nodes())]
        else:
            print("警告: 无法找到节点特征")
            return None
        
        # 过滤掉氢原子，只保留非氢原子
        non_h_atoms = []
        non_h_indices = []
        for i, atomic_num in enumerate(atomic_nums):
            if atomic_num != 1:  # 不是氢原子
                non_h_atoms.append(atomic_num)
                non_h_indices.append(i)
        
        if len(non_h_atoms) == 0:
            print("警告: 分子中只有氢原子")
            return None
        
        # 创建RDKit分子
        mol = Chem.RWMol()
        
        # 原子序数到原子符号的映射
        atom_map = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        
        # 添加非氢原子
        for atomic_num in non_h_atoms:
            if atomic_num in atom_map:
                atom = Chem.Atom(atom_map[atomic_num])
                mol.AddAtom(atom)
            else:
                print(f"警告: 未知的原子序数 {atomic_num}")
                return None
        
        # 提取化学键信息
        if 'edge_attr' in g.edata:
            # QM9数据集使用'edge_attr'作为边特征，前4维是化学键类型的one-hot编码
            edge_features = g.edata['edge_attr']
            bond_types = []
            for i in range(g.num_edges()):
                # 从one-hot编码中提取键类型
                bond_one_hot = edge_features[i][:4]  # 取前4维
                bond_type = torch.argmax(bond_one_hot).item()
                bond_types.append(bond_type)
        elif 'feat' in g.edata:
            # 备用方法：使用'feat'特征
            edge_features = g.edata['feat']
            bond_types = []
            for i in range(g.num_edges()):
                bond_one_hot = edge_features[i][:4]  # 取前4维
                bond_type = torch.argmax(bond_one_hot).item()
                bond_types.append(bond_type)
        else:
            print("警告: 无法找到边特征")
            return None
        
        # 化学键类型映射
        bond_map = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE, 3: Chem.BondType.AROMATIC}
        processed_bonds = set()
        
        # 创建原始索引到新索引的映射
        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
        
        # 添加化学键（只添加非氢原子之间的键）
        for i in range(g.num_edges()):
            src_node, dst_node = u[i].item(), v[i].item()
            bond_type_idx = bond_types[i]
            
            # 跳过包含氢原子的边
            if src_node not in old_to_new_idx or dst_node not in old_to_new_idx:
                continue
            
            # 转换为新的索引
            new_src = old_to_new_idx[src_node]
            new_dst = old_to_new_idx[dst_node]
            
            # 避免重复添加边（DGL图是双向的）
            edge_key = tuple(sorted([new_src, new_dst]))
            if edge_key in processed_bonds:
                continue
            
            if bond_type_idx in bond_map:
                bond_type = bond_map[bond_type_idx]
                mol.AddBond(new_src, new_dst, bond_type)
                processed_bonds.add(edge_key)
            else:
                print(f"警告: 未知的化学键类型 {bond_type_idx}")
        
        # 获取最终分子并清理
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol)
        return final_mol
        
    except Exception as e:
        print(f"从DGL图重建分子失败: {e}")
        return None

def generate_four_smiles_formats(mol: Chem.Mol) -> Dict[str, str]:
    """
    从RDKit分子生成四种SMILES格式
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        dict: 包含四种SMILES格式的字典
        - smiles_1: 重建后的分子直接转SMILES
        - smiles_2: 指定显示氢原子
        - smiles_3: 添加氢原子后转SMILES
        - smiles_4: 添加氢原子并指定显示
    """
    try:
        # 生成四种不同的SMILES格式
        smiles_1 = Chem.MolToSmiles(mol)  # 重建后的分子直接转SMILES
        smiles_2 = Chem.MolToSmiles(mol, allHsExplicit=True)  # 指定显示氢原子
        smiles_3 = Chem.MolToSmiles(Chem.AddHs(mol))  # 添加氢原子后转SMILES
        smiles_4 = Chem.MolToSmiles(Chem.AddHs(mol), allHsExplicit=True)  # 添加氢原子并指定显示
        
        return {
            'smiles_1': smiles_1,
            'smiles_2': smiles_2,
            'smiles_3': smiles_3,
            'smiles_4': smiles_4
        }
    except Exception as e:
        print(f"生成SMILES格式时出错: {e}")
        # 返回基本的SMILES作为fallback
        basic_smiles = Chem.MolToSmiles(mol)
        return {
            'smiles_1': basic_smiles,
            'smiles_2': basic_smiles,
            'smiles_3': basic_smiles,
            'smiles_4': basic_smiles
        }

def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """
    将数据集按照指定比例划分为训练集、验证集、测试集
    
    Args:
        data: 数据集列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"数据划分比例之和必须等于1.0，当前为{total_ratio}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算各集合的大小
    total_size = len(shuffled_data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # 划分数据
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    print(f"数据集划分完成:")
    print(f"  总数据量: {total_size}")
    print(f"  训练集: {len(train_data)} ({len(train_data)/total_size:.1%})")
    print(f"  验证集: {len(val_data)} ({len(val_data)/total_size:.1%})")
    print(f"  测试集: {len(test_data)} ({len(test_data)/total_size:.1%})")
    
    return train_data, val_data, test_data

def save_split_data(split_data: List[Dict], split_name: str, output_dir: str):
    """
    保存划分后的数据
    
    Args:
        split_data: 划分后的数据
        split_name: 划分名称 (train/val/test)
        output_dir: 输出目录
    """
    # 创建输出目录
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"\n保存 {split_name} 数据集...")
    
    # 提取图数据和SMILES
    graphs = []
    smiles_1_list = []
    smiles_2_list = []
    smiles_3_list = []
    smiles_4_list = []
    
    successful_conversions = 0
    failed_conversions = 0
    
    for i, sample in enumerate(split_data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(split_data)} (成功: {successful_conversions}, 失败: {failed_conversions})")
        
        # 获取图数据（原封不动）
        graph = sample['dgl_graph']
        label = sample['properties']  # 使用分子属性作为标签
        original_smiles = sample['smiles']  # 原始SMILES
        
        # 保存图数据（无论重建是否成功都保存）
        graphs.append((graph, label))
        
        # 从DGL图重建分子
        mol = dgl_graph_to_mol(graph)
        
        if mol is not None:
            # 生成四种SMILES格式
            four_smiles = generate_four_smiles_formats(mol)
            
            # 保存SMILES
            smiles_1_list.append(four_smiles['smiles_1'])
            smiles_2_list.append(four_smiles['smiles_2'])
            smiles_3_list.append(four_smiles['smiles_3'])
            smiles_4_list.append(four_smiles['smiles_4'])
            
            successful_conversions += 1
        else:
            # 重建失败，使用原始SMILES保存四遍
            smiles_1_list.append(original_smiles)
            smiles_2_list.append(original_smiles)
            smiles_3_list.append(original_smiles)
            smiles_4_list.append(original_smiles)
            
            failed_conversions += 1
            if failed_conversions <= 5:  # 只显示前5个失败案例
                print(f"  警告: 分子 {i} 重建失败，使用原始SMILES: {original_smiles}")
    
    print(f"  转换统计: 成功 {successful_conversions}, 失败 {failed_conversions}")
    print(f"  失败率: {failed_conversions/len(split_data)*100:.2f}%")
    
    # 保存图数据
    graph_file = os.path.join(split_dir, 'graphs.pkl')
    with open(graph_file, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"  图数据已保存到: {graph_file}")
    
    # 保存四种SMILES格式
    smiles_files = [
        ('smiles_1_direct.txt', smiles_1_list),
        ('smiles_2_explicit_h.txt', smiles_2_list),
        ('smiles_3_addhs.txt', smiles_3_list),
        ('smiles_4_addhs_explicit_h.txt', smiles_4_list)
    ]
    
    for filename, smiles_list in smiles_files:
        filepath = os.path.join(split_dir, filename)
        with open(filepath, 'w') as f:
            for smiles in smiles_list:
                f.write(smiles + '\n')
        print(f"  {filename} 已保存到: {filepath}")
    
    # 保存统计信息
    stats = {
        'split_name': split_name,
        'total_molecules': len(graphs),
        'successful_conversions': successful_conversions,
        'failed_conversions': failed_conversions,
        'failure_rate': failed_conversions/len(split_data),
        'graphs_file': 'graphs.pkl',
        'smiles_files': [filename for filename, _ in smiles_files],
        'data_format': 'DGL图 + 四种SMILES格式（从DGL图重建，失败时使用原始SMILES）'
    }
    
    stats_file = os.path.join(split_dir, 'split_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  统计信息已保存到: {stats_file}")

def main():
    """主函数"""
    print("QM9数据集处理脚本（从DGL图重建分子，忽略氢原子）")
    print("=" * 70)
    
    # 创建配置
    config = ProjectConfig()
    
    # 设置输出目录
    output_dir = "data/qm9_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 加载QM9数据集
    print("\n加载QM9数据集...")
    try:
        qm9_loader = QM9Loader(config)
        # 加载完整数据集（不限制数量）
        train_data, val_data, test_data, train_labels, val_labels, test_labels = qm9_loader.load_data(limit=None)
        # 合并所有数据
        dataset = train_data + val_data + test_data
        print(f"成功加载 {len(dataset)} 个分子")
    except Exception as e:
        print(f"加载QM9数据集失败: {e}")
        return
    
    # 验证数据格式
    print("\n验证数据格式...")
    if len(dataset) == 0:
        print("错误: 数据集为空")
        return
    
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"图类型: {type(sample['dgl_graph'])}")
    print(f"图节点数: {sample['dgl_graph'].num_nodes()}")
    print(f"图边数: {sample['dgl_graph'].num_edges()}")
    print(f"原始SMILES: {sample['smiles']}")
    print(f"属性数量: {len(sample['properties'])}")
    
    # 测试从DGL图重建分子
    print("\n测试从DGL图重建分子...")
    test_graph = sample['dgl_graph']
    test_mol = dgl_graph_to_mol(test_graph)
    if test_mol:
        print(f"✓ 测试分子重建成功: {Chem.MolToSmiles(test_mol)}")
        test_smiles = generate_four_smiles_formats(test_mol)
        print("四种SMILES格式:")
        for key, value in test_smiles.items():
            print(f"  {key}: {value}")
    else:
        print("✗ 测试分子重建失败，将使用原始SMILES")
    
    # 划分数据集
    print("\n划分数据集...")
    train_data, val_data, test_data = split_dataset(
        dataset, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1, 
        seed=42
    )
    
    # 保存各划分的数据
    print("\n保存数据集...")
    save_split_data(train_data, 'train', output_dir)
    save_split_data(val_data, 'val', output_dir)
    save_split_data(test_data, 'test', output_dir)
    
    # 保存总体统计信息
    overall_stats = {
        'dataset_name': 'QM9',
        'total_molecules': len(dataset),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'data_format': {
            'graphs': 'DGL图对象（原封不动）',
            'smiles_formats': [
                'smiles_1_direct.txt - 直接SMILES',
                'smiles_2_explicit_h.txt - 显式氢原子',
                'smiles_3_addhs.txt - 添加氢原子',
                'smiles_4_addhs_explicit_h.txt - 添加氢原子并显式'
            ],
            'smiles_source': '从DGL图重建分子生成，失败时使用原始SMILES',
            'hydrogen_handling': '重建时忽略氢原子和氢原子边'
        },
        'processing_info': {
            'loader': 'QM9Loader',
            'molecule_reconstruction': 'dgl_graph_to_mol (忽略氢原子)',
            'smiles_conversion': 'generate_four_smiles_formats',
            'split_method': 'random_split',
            'split_ratios': {'train': 0.8, 'val': 0.1, 'test': 0.1},
            'fallback_strategy': '重建失败时使用原始SMILES保存四遍'
        }
    }
    
    overall_stats_file = os.path.join(output_dir, 'overall_stats.json')
    with open(overall_stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"\n总体统计信息已保存到: {overall_stats_file}")
    
    print("\n" + "=" * 70)
    print("✓ QM9数据集处理完成！")
    print(f"✓ 输出目录: {output_dir}")
    print("✓ 数据格式: DGL图 + 四种SMILES格式")
    print("✓ 数据集划分: train/val/test")
    print("✓ 数据集数量: 保持全量QM9数据集")
    print("✓ 氢原子处理: 重建时忽略氢原子")
    print("✓ 失败处理: 重建失败时使用原始SMILES")
    print("✓ 所有文件已按相同顺序保存")

if __name__ == "__main__":
    main() 
