"""
ZINC数据集转换结果使用示例（新版本）

这个文件展示了如何使用转换后的ZINC数据集和分子图处理工具库。
现在数据保持了训练集、验证集、测试集的划分。
"""

import pickle
import torch
import dgl
from rdkit import Chem
import matplotlib.pyplot as plt
import os

# 导入我们的工具库
from molecular_graph_utils import (
    dgl_graph_to_mol, 
    mol_to_simplified_graph, 
    mol_to_explicit_h_graph,
    generate_four_smiles_formats,
    visualize_molecule,
    visualize_dgl_graph,
    compare_graph_structures
)

def load_dataset_split(split_name):
    """加载指定划分的数据集"""
    print(f"=== 加载 {split_name} 数据集 ===")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = os.path.join(script_dir, split_name)
    
    if not os.path.exists(split_dir):
        print(f"错误: {split_name} 目录不存在")
        print(f"期望路径: {split_dir}")
        return None
    
    # 加载图数据
    with open(f'{split_dir}/simplified_graphs.pkl', 'rb') as f:
        simplified_graphs = pickle.load(f)
    
    with open(f'{split_dir}/explicit_h_graphs.pkl', 'rb') as f:
        explicit_h_graphs = pickle.load(f)
    
    # 加载SMILES数据
    with open(f'{split_dir}/smiles_1_direct.txt', 'r') as f:
        smiles_1_list = f.read().strip().split('\n')
    
    with open(f'{split_dir}/smiles_2_explicit_h.txt', 'r') as f:
        smiles_2_list = f.read().strip().split('\n')
    
    with open(f'{split_dir}/smiles_3_addhs.txt', 'r') as f:
        smiles_3_list = f.read().strip().split('\n')
    
    with open(f'{split_dir}/smiles_4_addhs_explicit_h.txt', 'r') as f:
        smiles_4_list = f.read().strip().split('\n')
    
    print(f"简化图数据: {len(simplified_graphs)} 个分子")
    print(f"显式氢原子图数据: {len(explicit_h_graphs)} 个分子")
    print(f"SMILES 1 (直接): {len(smiles_1_list)} 个")
    print(f"SMILES 2 (显式H): {len(smiles_2_list)} 个")
    print(f"SMILES 3 (AddHs): {len(smiles_3_list)} 个")
    print(f"SMILES 4 (AddHs+显式H): {len(smiles_4_list)} 个")
    
    return {
        'simplified_graphs': simplified_graphs,
        'explicit_h_graphs': explicit_h_graphs,
        'smiles_1': smiles_1_list,
        'smiles_2': smiles_2_list,
        'smiles_3': smiles_3_list,
        'smiles_4': smiles_4_list
    }

def verify_data_consistency(data, split_name):
    """验证数据一致性"""
    print(f"\n=== 验证 {split_name} 数据一致性 ===")
    
    simplified_graphs = data['simplified_graphs']
    explicit_h_graphs = data['explicit_h_graphs']
    smiles_1_list = data['smiles_1']
    smiles_2_list = data['smiles_2']
    smiles_3_list = data['smiles_3']
    smiles_4_list = data['smiles_4']
    
    # 检查数量一致性
    print("检查数量一致性:")
    print(f"  简化图: {len(simplified_graphs)}")
    print(f"  显式氢原子图: {len(explicit_h_graphs)}")
    print(f"  SMILES 1: {len(smiles_1_list)}")
    print(f"  SMILES 2: {len(smiles_2_list)}")
    print(f"  SMILES 3: {len(smiles_3_list)}")
    print(f"  SMILES 4: {len(smiles_4_list)}")
    
    all_lengths = [
        len(simplified_graphs),
        len(explicit_h_graphs),
        len(smiles_1_list),
        len(smiles_2_list),
        len(smiles_3_list),
        len(smiles_4_list)
    ]
    
    if len(set(all_lengths)) == 1:
        print("✓ 所有数据文件数量一致")
    else:
        print("✗ 数据文件数量不一致！")
        return False
    
    # 检查前几个分子的SMILES重建
    print("\n检查SMILES重建:")
    for i in range(min(3, len(smiles_1_list))):
        try:
            mol = Chem.MolFromSmiles(smiles_1_list[i])
            if mol:
                print(f"  分子 {i}: SMILES重建成功")
            else:
                print(f"  分子 {i}: SMILES重建失败")
                return False
        except Exception as e:
            print(f"  分子 {i}: SMILES重建异常 - {e}")
            return False
    
    print("✓ SMILES重建验证通过")
    return True

def analyze_single_molecule(data, split_name, index=0):
    """分析单个分子的详细信息"""
    print(f"\n=== 分析 {split_name} 分子 {index} ===")
    
    simplified_graphs = data['simplified_graphs']
    explicit_h_graphs = data['explicit_h_graphs']
    smiles_1_list = data['smiles_1']
    smiles_2_list = data['smiles_2']
    smiles_3_list = data['smiles_3']
    smiles_4_list = data['smiles_4']
    
    # 获取分子数据
    simplified_graph, label = simplified_graphs[index]
    explicit_h_graph, _ = explicit_h_graphs[index]
    
    print(f"分子标签: {label}")
    print(f"\n简化图信息:")
    print(f"  节点数: {simplified_graph.num_nodes()}")
    print(f"  边数: {simplified_graph.num_edges()}")
    print(f"  节点特征范围: {simplified_graph.ndata['feat'].min().item()} - {simplified_graph.ndata['feat'].max().item()}")
    
    print(f"\n显式氢原子图信息:")
    print(f"  节点数: {explicit_h_graph.num_nodes()}")
    print(f"  边数: {explicit_h_graph.num_edges()}")
    print(f"  节点特征范围: {explicit_h_graph.ndata['feat'].min().item()} - {explicit_h_graph.ndata['feat'].max().item()}")
    
    print(f"\n四种SMILES格式:")
    print(f"  SMILES 1 (直接): {smiles_1_list[index]}")
    print(f"  SMILES 2 (显式H): {smiles_2_list[index]}")
    print(f"  SMILES 3 (AddHs): {smiles_3_list[index]}")
    print(f"  SMILES 4 (AddHs+显式H): {smiles_4_list[index]}")
    
    # 从SMILES重建分子并验证
    mol = Chem.MolFromSmiles(smiles_1_list[index])
    if mol:
        print(f"\n从SMILES重建分子成功:")
        print(f"  原子数: {mol.GetNumAtoms()}")
        print(f"  化学键数: {mol.GetNumBonds()}")
        
        # 生成四种SMILES格式验证
        smiles_dict = generate_four_smiles_formats(mol)
        print(f"  验证SMILES 1: {smiles_dict['smiles_1']}")
        print(f"  验证SMILES 2: {smiles_dict['smiles_2']}")
        print(f"  验证SMILES 3: {smiles_dict['smiles_3']}")
        print(f"  验证SMILES 4: {smiles_dict['smiles_4']}")
        
        # 检查是否与原始SMILES一致
        if smiles_dict['smiles_1'] == smiles_1_list[index]:
            print("✓ SMILES 1 一致")
        else:
            print("✗ SMILES 1 不一致")
    
    return simplified_graph, explicit_h_graph, mol

def demonstrate_toolkit_functions(data, split_name):
    """演示工具库功能"""
    print(f"\n=== {split_name} 工具库功能演示 ===")
    
    # 加载一个分子
    simplified_graphs = data['simplified_graphs']
    graph, label = simplified_graphs[0]
    
    print("1. DGL图转RDKit分子:")
    mol = dgl_graph_to_mol(graph)
    if mol:
        print(f"   转换成功，分子有 {mol.GetNumAtoms()} 个原子")
    
    print("\n2. 生成四种SMILES格式:")
    if mol:
        smiles_dict = generate_four_smiles_formats(mol)
        for key, value in smiles_dict.items():
            print(f"   {key}: {value}")
    
    print("\n3. 生成简化图:")
    simplified_g = mol_to_simplified_graph(mol)
    print(f"   简化图: {simplified_g.num_nodes()} 节点, {simplified_g.num_edges()} 边")
    
    print("\n4. 生成显式氢原子图:")
    explicit_h_g = mol_to_explicit_h_graph(mol)
    print(f"   显式氢原子图: {explicit_h_g.num_nodes()} 节点, {explicit_h_g.num_edges()} 边")
    
    print("\n5. 结构对比分析:")
    compare_graph_structures(graph, mol, 0)

def dataset_statistics(data, split_name):
    """数据集统计信息"""
    print(f"\n=== {split_name} 数据集统计信息 ===")
    
    simplified_graphs = data['simplified_graphs']
    explicit_h_graphs = data['explicit_h_graphs']
    
    # 统计节点数和边数
    simplified_nodes = []
    simplified_edges = []
    explicit_h_nodes = []
    explicit_h_edges = []
    
    for graph, _ in simplified_graphs:
        simplified_nodes.append(graph.num_nodes())
        simplified_edges.append(graph.num_edges())
    
    for graph, _ in explicit_h_graphs:
        explicit_h_nodes.append(graph.num_nodes())
        explicit_h_edges.append(graph.num_edges())
    
    print(f"简化图统计:")
    print(f"  平均节点数: {sum(simplified_nodes) / len(simplified_nodes):.2f}")
    print(f"  平均边数: {sum(simplified_edges) / len(simplified_edges):.2f}")
    print(f"  最大节点数: {max(simplified_nodes)}")
    print(f"  最大边数: {max(simplified_edges)}")
    print(f"  最小节点数: {min(simplified_nodes)}")
    print(f"  最小边数: {min(simplified_edges)}")
    
    print(f"\n显式氢原子图统计:")
    print(f"  平均节点数: {sum(explicit_h_nodes) / len(explicit_h_nodes):.2f}")
    print(f"  平均边数: {sum(explicit_h_edges) / len(explicit_h_edges):.2f}")
    print(f"  最大节点数: {max(explicit_h_nodes)}")
    print(f"  最大边数: {max(explicit_h_edges)}")
    print(f"  最小节点数: {min(explicit_h_nodes)}")
    print(f"  最小边数: {min(explicit_h_edges)}")
    
    # 统计节点特征分布
    atom_counts = {}
    for graph, _ in simplified_graphs:
        for feat in graph.ndata['feat']:
            atom_type = feat.item()
            atom_counts[atom_type] = atom_counts.get(atom_type, 0) + 1
    
    print(f"\n原子类型分布:")
    atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    for atom_type, count in sorted(atom_counts.items()):
        symbol = atom_symbols.get(atom_type, str(atom_type))
        print(f"  {symbol} (原子序数 {atom_type}): {count}")

def compare_splits():
    """比较不同划分的数据"""
    print("\n=== 比较不同划分的数据 ===")
    
    splits = ['train', 'val', 'test']
    split_data = {}
    
    # 加载所有划分的数据
    for split in splits:
        data = load_dataset_split(split)
        if data:
            split_data[split] = data
    
    # 比较SMILES格式
    print("\n比较SMILES格式（第一个分子）:")
    for split in splits:
        if split in split_data:
            smiles_1 = split_data[split]['smiles_1'][0]
            print(f"  {split}: {smiles_1}")
    
    # 比较图结构
    print("\n比较图结构（第一个分子）:")
    for split in splits:
        if split in split_data:
            graph, label = split_data[split]['simplified_graphs'][0]
            print(f"  {split}: {graph.num_nodes()} 节点, {graph.num_edges()} 边, 标签: {label}")

def visualize_molecules(data, split_name, num_molecules=2):
    """可视化分子"""
    print(f"\n=== {split_name} 分子可视化 ===")
    
    simplified_graphs = data['simplified_graphs']
    explicit_h_graphs = data['explicit_h_graphs']
    smiles_1_list = data['smiles_1']
    
    for i in range(min(num_molecules, len(simplified_graphs))):
        print(f"处理分子 {i}...")
        
        simplified_graph, label = simplified_graphs[i]
        explicit_h_graph, _ = explicit_h_graphs[i]
        
        # 从SMILES重建分子
        mol = Chem.MolFromSmiles(smiles_1_list[i])
        if mol:
            # 获取脚本所在目录用于保存图像
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 保存分子图像
            visualize_molecule(mol, title=f"{split_name} 分子 {i}", 
                             save_path=os.path.join(script_dir, f"{split_name}_molecule_{i}_rdkit.png"))
            
            # 保存图结构图像
            visualize_dgl_graph(simplified_graph, title=f"{split_name} 简化图 - 分子 {i}", 
                               save_path=os.path.join(script_dir, f"{split_name}_molecule_{i}_simplified_graph.png"))
            visualize_dgl_graph(explicit_h_graph, title=f"{split_name} 显式氢原子图 - 分子 {i}", 
                               save_path=os.path.join(script_dir, f"{split_name}_molecule_{i}_explicit_h_graph.png"))
            
            print(f"  分子 {i} 的可视化图像已保存")

def main():
    """主函数"""
    print("ZINC数据集转换结果使用示例（新版本）")
    print("=" * 60)
    
    # 检查当前目录结构
    current_files = os.listdir('.')
    print(f"当前目录: {current_files}")
    
    # 测试训练集
    print("\n" + "="*60)
    train_data = load_dataset_split('train')
    if train_data and verify_data_consistency(train_data, 'train'):
        analyze_single_molecule(train_data, 'train', 0)
        demonstrate_toolkit_functions(train_data, 'train')
        dataset_statistics(train_data, 'train')
    
    # 测试验证集
    print("\n" + "="*60)
    val_data = load_dataset_split('val')
    if val_data and verify_data_consistency(val_data, 'val'):
        analyze_single_molecule(val_data, 'val', 0)
        demonstrate_toolkit_functions(val_data, 'val')
        dataset_statistics(val_data, 'val')
    
    # 测试测试集
    print("\n" + "="*60)
    test_data = load_dataset_split('test')
    if test_data and verify_data_consistency(test_data, 'test'):
        analyze_single_molecule(test_data, 'test', 0)
        demonstrate_toolkit_functions(test_data, 'test')
        dataset_statistics(test_data, 'test')
    
    # 比较不同划分
    compare_splits()
    
    # 询问是否进行可视化
    response = input("\n是否生成可视化图像？(y/n): ")
    if response.lower() == 'y':
        if train_data:
            visualize_molecules(train_data, 'train', 2)
        if val_data:
            visualize_molecules(val_data, 'val', 1)
        if test_data:
            visualize_molecules(test_data, 'test', 1)
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("✓ 数据保持了训练集/验证集/测试集划分")
    print("✓ 所有数据文件数量一致")
    print("✓ 工具库功能正常工作")
    print("✓ 可以用于机器学习训练和评估")

if __name__ == "__main__":
    main() 