"""
分子图处理工具库

这个库提供了将不同格式的分子数据转换为标准化图格式的功能。
主要包含以下功能：
1. DGL图到RDKit分子的转换
2. RDKit分子到标准化图的转换
3. 多种SMILES格式生成
4. 分子图可视化
5. 结构对比分析

作者: AI Assistant
日期: 2024
"""

import torch
import dgl
import re
import pickle
import os
import sys
import json
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# --- 常量定义 ---

# ZINC数据集的原子类型映射
ZINC_ATOM_MAP = {
    'C': 0, 'O': 1, 'N': 2, 'F': 3, 'C H1': 4, 'S': 5, 'Cl': 6, 'O -': 7,
    'N H1 +': 8, 'Br': 9, 'N H3 +': 10, 'N H2 +': 11, 'N +': 12, 'N -': 13,
    'S -': 14, 'I': 15, 'P': 16, 'O H1 +': 17, 'N H1 -': 18, 'O +': 19,
    'S +': 20, 'P H1': 21, 'P H2': 22, 'C H2 -': 23, 'P +': 24,
    'S H1 +': 25, 'C H1 -': 26, 'P H1 +': 27
}

# 通用化学键类型映射
BOND_MAP = {'NONE': 0, 'SINGLE': 1, 'DOUBLE': 2, 'TRIPLE': 3, 'AROMATIC': 4}

# 原子序数映射
ATOMIC_NUMBERS = {
    'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'H': 1
}

# --- 核心转换函数 ---

def parse_atom_feature(feature_str: str) -> Tuple[str, int, int]:
    """
    解析原子特征字符串，返回原子符号、氢原子数和电荷
    
    Args:
        feature_str: 原子特征字符串，如 "C H1", "N H1 +", "O -"
    
    Returns:
        Tuple[str, int, int]: (原子符号, 氢原子数, 电荷)
    """
    match = re.match(r'([A-Z][a-z]?)', feature_str)
    symbol = match.group(1)
    h_match = re.search(r'H(\d)', feature_str)
    num_explicit_hs = int(h_match.group(1)) if h_match else 0
    charge = 0
    if '+' in feature_str:
        charge = feature_str.count('+')
    elif '-' in feature_str:
        charge = -feature_str.count('-')
    return symbol, num_explicit_hs, charge

def dgl_graph_to_mol(g: dgl.DGLGraph, atom_map: Dict[str, int] = None) -> Optional[Chem.Mol]:
    """
    将DGL图对象转换为RDKit分子对象
    
    Args:
        g: DGL图对象
        atom_map: 原子类型映射字典，默认为ZINC_ATOM_MAP
    
    Returns:
        RDKit分子对象，转换失败返回None
    """
    if atom_map is None:
        atom_map = ZINC_ATOM_MAP
    
    atom_map_rev = {v: k for k, v in atom_map.items()}
    bond_map_rev = {v: k for k, v in BOND_MAP.items()}
    
    mol = Chem.RWMol()

    # 添加原子
    node_features = g.ndata['feat']
    for feature_idx in node_features:
        feature_str = atom_map_rev.get(feature_idx.item())
        if feature_str is None:
            print(f"警告: 未知的原子特征索引 {feature_idx.item()}")
            return None
        symbol, num_hs, charge = parse_atom_feature(feature_str)
        atom = Chem.Atom(symbol)
        atom.SetNumExplicitHs(num_hs)
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)

    # 添加化学键
    u, v = g.edges()
    edge_features = g.edata['feat']
    processed_bonds = set()
    for i in range(g.num_edges()):
        src_node, dst_node = u[i].item(), v[i].item()
        if tuple(sorted((src_node, dst_node))) in processed_bonds:
            continue
        bond_feature_idx = edge_features[i].item()
        bond_str = bond_map_rev.get(bond_feature_idx)
        if bond_str is None or bond_str == 'NONE':
            continue
        
        # 转换化学键类型
        if bond_str == 'SINGLE':
            bond_type = Chem.BondType.SINGLE
        elif bond_str == 'DOUBLE':
            bond_type = Chem.BondType.DOUBLE
        elif bond_str == 'TRIPLE':
            bond_type = Chem.BondType.TRIPLE
        elif bond_str == 'AROMATIC':
            bond_type = Chem.BondType.AROMATIC
        else:
            continue
            
        mol.AddBond(src_node, dst_node, bond_type)
        processed_bonds.add(tuple(sorted((src_node, dst_node))))

    try:
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol)
        return final_mol
    except Exception as e:
        print(f"RDKit转换失败: {e}")
        return None

def mol_to_simplified_graph(mol: Chem.Mol) -> dgl.DGLGraph:
    """
    将RDKit分子转换为简化的图（使用原子序数作为节点特征）
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        DGL图对象
    """
    # 添加节点（原子序数特征）
    num_atoms = mol.GetNumAtoms()
    node_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        node_features.append(atomic_num)
    
    # 准备边数据
    src_nodes = []
    dst_nodes = []
    edge_features = []
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        # 转换化学键类型
        if bond_type == Chem.BondType.SINGLE:
            bond_feature = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_feature = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_feature = 3
        elif bond_type == Chem.BondType.AROMATIC:
            bond_feature = 4
        else:
            bond_feature = 0
            print(f"警告: 未知的化学键类型 {bond_type}")
        
        # 添加双向边
        src_nodes.extend([begin_idx, end_idx])
        dst_nodes.extend([end_idx, begin_idx])
        edge_features.extend([bond_feature, bond_feature])
    
    # 创建DGL图
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.long)
    g.edata['feat'] = torch.tensor(edge_features, dtype=torch.long)
    
    return g

def mol_to_explicit_h_graph(mol: Chem.Mol) -> dgl.DGLGraph:
    """
    将RDKit分子转换为显式氢原子图
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        DGL图对象（包含显式氢原子）
    """
    # 创建包含显式氢原子的分子
    explicit_mol = Chem.AddHs(mol)
    
    # 添加节点（原子序数特征）
    num_atoms = explicit_mol.GetNumAtoms()
    node_features = []
    for atom in explicit_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        node_features.append(atomic_num)
    
    # 准备边数据
    src_nodes = []
    dst_nodes = []
    edge_features = []
    
    for bond in explicit_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        
        # 转换化学键类型
        if bond_type == Chem.BondType.SINGLE:
            bond_feature = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_feature = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_feature = 3
        elif bond_type == Chem.BondType.AROMATIC:
            bond_feature = 4
        else:
            bond_feature = 0
            print(f"警告: 未知的化学键类型 {bond_type}")
        
        # 添加双向边
        src_nodes.extend([begin_idx, end_idx])
        dst_nodes.extend([end_idx, begin_idx])
        edge_features.extend([bond_feature, bond_feature])
    
    # 创建DGL图
    g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_atoms)
    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.long)
    g.edata['feat'] = torch.tensor(edge_features, dtype=torch.long)
    
    return g

def generate_four_smiles_formats(mol: Chem.Mol) -> Dict[str, str]:
    """
    生成分子的四种SMILES表示格式
    
    Args:
        mol: RDKit分子对象
    
    Returns:
        dict: 包含四种SMILES格式的字典
        - smiles_1: 重建后的分子直接转SMILES
        - smiles_2: 指定显示氢原子
        - smiles_3: 添加氢原子后转SMILES
        - smiles_4: 添加氢原子并指定显示
    """
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

# --- 可视化函数 ---

def visualize_molecule(mol: Chem.Mol, title: str = "", save_path: Optional[str] = None):
    """
    可视化RDKit分子对象
    
    Args:
        mol: RDKit分子对象
        title: 图像标题
        save_path: 保存路径，None则显示图像
    """
    img = MolToImage(mol, size=(300, 300))
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"分子图像已保存到: {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_dgl_graph(g: dgl.DGLGraph, title: str = "", save_path: Optional[str] = None):
    """
    可视化DGL图结构
    
    Args:
        g: DGL图对象
        title: 图像标题
        save_path: 保存路径，None则显示图像
    """
    plt.figure(figsize=(10, 8))
    
    # 获取图的节点和边信息
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    
    # 获取节点特征（原子类型）
    node_features = g.ndata['feat']
    
    # 获取边特征（化学键类型）
    edge_features = g.edata['feat']
    u, v = g.edges()
    
    # 创建更好的节点布局（使用spring布局的简化版本）
    # 创建邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(g.num_edges()):
        src, dst = u[i].item(), v[i].item()
        adj_matrix[src, dst] = 1
        adj_matrix[dst, src] = 1
    
    # 使用简单的力导向布局
    pos = np.random.rand(num_nodes, 2) * 2 - 1  # 随机初始位置
    
    # 简单的力导向迭代
    for _ in range(50):
        new_pos = pos.copy()
        for i in range(num_nodes):
            force = np.zeros(2)
            for j in range(num_nodes):
                if i != j:
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        if adj_matrix[i, j] > 0:  # 连接的节点相互吸引
                            force += diff * 0.1 / dist
                        else:  # 未连接的节点相互排斥
                            force -= diff * 0.05 / (dist**2)
            new_pos[i] += force * 0.1
        pos = new_pos
    
    # 绘制边（化学键）
    bond_colors = {1: 'black', 2: 'red', 3: 'blue', 4: 'green'}  # 单键、双键、三键、芳香键
    bond_widths = {1: 1, 2: 2, 3: 3, 4: 1}
    
    for i in range(g.num_edges()):
        src, dst = u[i].item(), v[i].item()
        bond_type = edge_features[i].item()
        color = bond_colors.get(bond_type, 'gray')
        width = bond_widths.get(bond_type, 1)
        plt.plot([pos[src, 0], pos[dst, 0]], [pos[src, 1], pos[dst, 1]], 
                color=color, linewidth=width, alpha=0.7)
    
    # 绘制节点（原子）
    atom_colors = {
        1: 'white',      # H
        6: 'lightgray',  # C
        7: 'blue',       # N
        8: 'red',        # O
        9: 'green',      # F
        15: 'brown',     # P
        16: 'yellow',    # S
        17: 'orange',    # Cl
        35: 'darkred',   # Br
        53: 'purple'     # I
    }
    
    for i in range(num_nodes):
        atom_type = node_features[i].item()
        color = atom_colors.get(atom_type, 'lightblue')
        
        # 获取原子符号
        atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        symbol = atom_symbols.get(atom_type, str(atom_type))
        
        plt.scatter(pos[i, 0], pos[i, 1], c=color, s=200, zorder=3, edgecolors='black', linewidth=1)
        plt.annotate(symbol, (pos[i, 0], pos[i, 1]), xytext=(0, 0), 
                    textcoords='offset points', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    plt.title(f"{title}\nNodes: {num_nodes}, Edges: {num_edges}")
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=1, label='Single Bond'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Double Bond'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Triple Bond'),
        plt.Line2D([0], [0], color='green', linewidth=1, label='Aromatic Bond')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"DGL图结构已保存到: {save_path}")
    else:
        plt.show()
    plt.close()

# --- 分析函数 ---

def compare_graph_structures(dgl_graph: dgl.DGLGraph, rdkit_mol: Chem.Mol, molecule_index: int = 0):
    """
    详细比较DGL图和RDKit分子的结构
    
    Args:
        dgl_graph: DGL图对象
        rdkit_mol: RDKit分子对象
        molecule_index: 分子索引（用于打印）
    """
    print(f"\n=== 分子 {molecule_index} 结构对比分析 ===")
    
    # DGL图信息
    dgl_nodes = dgl_graph.num_nodes()
    dgl_edges = dgl_graph.num_edges()
    dgl_node_features = dgl_graph.ndata['feat']
    dgl_edge_features = dgl_graph.edata['feat']
    
    # RDKit分子信息
    rdkit_atoms = rdkit_mol.GetNumAtoms()
    rdkit_bonds = rdkit_mol.GetNumBonds()
    
    print(f"DGL图结构:")
    print(f"  - 节点数: {dgl_nodes}")
    print(f"  - 边数: {dgl_edges}")
    
    # 检查DGL图是否为双向图
    u, v = dgl_graph.edges()
    unique_edges = set()
    for i in range(dgl_graph.num_edges()):
        src, dst = u[i].item(), v[i].item()
        # 将边标准化为有序对（较小的节点在前）
        edge = tuple(sorted([src, dst]))
        unique_edges.add(edge)
    dgl_unique_edges = len(unique_edges)
    
    print(f"  - 唯一边数（去重后）: {dgl_unique_edges}")
    print(f"  - 是否为双向图: {'是' if dgl_edges == 2 * dgl_unique_edges else '否'}")
    
    print(f"  - 原子类型分布:")
    atom_counts = {}
    for feat in dgl_node_features:
        atom_type = feat.item()
        atom_counts[atom_type] = atom_counts.get(atom_type, 0) + 1
    for atom_type, count in atom_counts.items():
        print(f"    {atom_type}: {count}")
    
    print(f"  - 化学键类型分布:")
    bond_counts = {}
    for feat in dgl_edge_features:
        bond_type = feat.item()
        bond_counts[bond_type] = bond_counts.get(bond_type, 0) + 1
    for bond_type, count in bond_counts.items():
        print(f"    {bond_type}: {count}")
    
    print(f"\nRDKit分子结构:")
    print(f"  - 原子数: {rdkit_atoms}")
    print(f"  - 化学键数: {rdkit_bonds}")
    print(f"  - 原子类型分布:")
    rdkit_atom_counts = {}
    for atom in rdkit_mol.GetAtoms():
        symbol = atom.GetSymbol()
        rdkit_atom_counts[symbol] = rdkit_atom_counts.get(symbol, 0) + 1
    for symbol, count in rdkit_atom_counts.items():
        print(f"    {symbol}: {count}")
    
    print(f"  - 化学键类型分布:")
    rdkit_bond_counts = {}
    for bond in rdkit_mol.GetBonds():
        bond_type = str(bond.GetBondType())
        rdkit_bond_counts[bond_type] = rdkit_bond_counts.get(bond_type, 0) + 1
    for bond_type, count in rdkit_bond_counts.items():
        print(f"    {bond_type}: {count}")
    
    # 结构一致性检查
    print(f"\n结构一致性检查:")
    if dgl_nodes == rdkit_atoms:
        print(f"  ✓ 原子数量一致: {dgl_nodes}")
    else:
        print(f"  ✗ 原子数量不一致: DGL={dgl_nodes}, RDKit={rdkit_atoms}")
    
    # 比较去重后的边数与RDKit化学键数
    if dgl_unique_edges == rdkit_bonds:
        print(f"  ✓ 化学键数量一致（考虑双向边）: DGL去重后={dgl_unique_edges}, RDKit={rdkit_bonds}")
    else:
        print(f"  ✗ 化学键数量不一致: DGL去重后={dgl_unique_edges}, RDKit={rdkit_bonds}")
    
    # 检查是否有结构变化
    structure_changed = (dgl_nodes != rdkit_atoms) or (dgl_unique_edges != rdkit_bonds)
    if structure_changed:
        print(f"  ⚠️  检测到结构变化！转换过程中分子结构发生了改变。")
    else:
        print(f"  ✓ 结构完全一致，转换过程中没有结构变化。")

# --- 数据处理函数 ---

def process_molecule_dataset(dataset, output_dir: str = "converted_data"):
    """
    处理分子数据集，生成标准化的图数据和SMILES
    
    Args:
        dataset: 包含(graph, label)元组的数据集
        output_dir: 输出目录
    
    Returns:
        dict: 处理统计信息
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 存储结果
    simplified_graphs = []
    explicit_h_graphs = []
    smiles_1_list = []
    smiles_2_list = []
    smiles_3_list = []
    smiles_4_list = []
    
    print(f"处理数据集 ({len(dataset)} 个分子)...")
    for i, (dgl_graph, label) in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(dataset)}")
        
        # 转换为RDKit分子
        mol = dgl_graph_to_mol(dgl_graph)
        if mol is None:
            continue
        
        # 生成简化图
        simplified_g = mol_to_simplified_graph(mol)
        simplified_graphs.append((simplified_g, label))
        
        # 生成显式氢原子图
        explicit_h_g = mol_to_explicit_h_graph(mol)
        explicit_h_graphs.append((explicit_h_g, label))
        
        # 生成四种不同的SMILES字符串
        smiles_dict = generate_four_smiles_formats(mol)
        
        smiles_1_list.append(smiles_dict['smiles_1'])
        smiles_2_list.append(smiles_dict['smiles_2'])
        smiles_3_list.append(smiles_dict['smiles_3'])
        smiles_4_list.append(smiles_dict['smiles_4'])
    
    # 保存结果
    print("保存转换结果...")
    
    # 保存图数据
    simplified_data_path = os.path.join(output_dir, "simplified_graphs.pkl")
    with open(simplified_data_path, 'wb') as f:
        pickle.dump(simplified_graphs, f)
    print(f"简化图数据已保存到: {simplified_data_path}")
    
    explicit_h_data_path = os.path.join(output_dir, "explicit_h_graphs.pkl")
    with open(explicit_h_data_path, 'wb') as f:
        pickle.dump(explicit_h_graphs, f)
    print(f"显式氢原子图数据已保存到: {explicit_h_data_path}")
    
    # 保存SMILES文件
    smiles_files = [
        ("smiles_1_direct.txt", smiles_1_list),
        ("smiles_2_explicit_h.txt", smiles_2_list),
        ("smiles_3_addhs.txt", smiles_3_list),
        ("smiles_4_addhs_explicit_h.txt", smiles_4_list)
    ]
    
    for filename, smiles_list in smiles_files:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            for smiles in smiles_list:
                f.write(smiles + '\n')
        print(f"{filename} 已保存到: {filepath}")
    
    # 保存统计信息
    stats = {
        "total_molecules": len(simplified_graphs),
        "simplified_graphs": len(simplified_graphs),
        "explicit_h_graphs": len(explicit_h_graphs),
        "smiles_1_direct": len(smiles_1_list),
        "smiles_2_explicit_h": len(smiles_2_list),
        "smiles_3_addhs": len(smiles_3_list),
        "smiles_4_addhs_explicit_h": len(smiles_4_list)
    }
    
    stats_path = os.path.join(output_dir, "conversion_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"统计信息已保存到: {stats_path}")
    
    print(f"\n转换完成！共处理了 {len(simplified_graphs)} 个分子。")
    return stats

# --- 使用示例 ---

if __name__ == "__main__":
    print("分子图处理工具库")
    print("=" * 50)
    print("主要功能:")
    print("1. dgl_graph_to_mol() - DGL图转RDKit分子")
    print("2. mol_to_simplified_graph() - 生成简化图")
    print("3. mol_to_explicit_h_graph() - 生成显式氢原子图")
    print("4. generate_four_smiles_formats() - 生成四种SMILES格式")
    print("5. visualize_molecule() - 分子可视化")
    print("6. visualize_dgl_graph() - 图结构可视化")
    print("7. compare_graph_structures() - 结构对比分析")
    print("8. process_molecule_dataset() - 批量处理数据集")
    print("\n详细使用说明请参考文档。") 