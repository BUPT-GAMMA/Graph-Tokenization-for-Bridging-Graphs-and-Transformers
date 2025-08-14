#!/usr/bin/env python3
"""
测试不同序列化方法
从qm9test数据集中随机抽取分子，对每个分子使用不同的序列化方法，并打印原始结构
"""

import sys
import os
import random
import numpy as np
from typing import List, Dict, Any
from rdkit.Chem import AllChem

from src.data.base_loader import BaseDataLoader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from src.algorithms.serializer.serializer_factory import SerializerFactory
from src.utils import get_logger

logger = get_logger(__name__)

def print_molecule_info(sample: Dict[str, Any], sample_idx: int):
    """打印分子信息"""
    print(f"\n{'='*80}")
    print(f"📊 分子 {sample_idx + 1}")
    print(f"{'='*80}")
    
    # 基本信息
    print(f"🆔 分子ID: {sample.get('id', 'unknown')}")
    print(f"📝 SMILES: {sample.get('smiles', 'N/A')}")
    print(f"🔢 节点数: {sample.get('num_nodes', 0)}")
    print(f"🔗 边数: {sample.get('num_edges', 0)}")
    
    # 分子属性
    # properties = sample.get('properties', {})
    # if properties:
    #     print(f"📈 分子属性:")
    #     for prop_name, prop_value in list(properties):  # 只显示前5个属性
    #         print(f"   {prop_name}: {prop_value:.4f}")
    #     if len(properties) > 5:
    #         print(f"   ... 还有 {len(properties) - 5} 个属性")
    
    # DGL图信息
    graph = sample.get('dgl_graph')
    if graph is not None:
        print(f"📊 DGL图信息:")
        print(f"   节点特征维度: {list(graph.ndata.keys())}")
        print(f"   边特征维度: {list(graph.edata.keys())}")
        
        # 显示节点特征示例
        if 'feat' in graph.ndata:
            node_feat = graph.ndata['feat']
            print(f"   节点特征形状: {node_feat.shape}")
            print(f"   节点特征示例 (前3个节点):")
            for i in range(min(3, node_feat.shape[0])):
                feat = node_feat
                print(f"     节点{i}: {feat.tolist()}...")  # 只显示前10维
        
        # 显示边特征示例
        if 'feat' in graph.edata:
            edge_feat = graph.edata['feat']
            print(f"   边特征形状: {edge_feat.shape}")
            print(f"   边特征示例 (前3条边):")
            for i in range(min(3, edge_feat.shape[0])):
                feat = edge_feat
                print(f"     边{i}: {feat.tolist()}")

def test_serialization_methods(sample: Dict[str, Any], methods: List[str],dataloader: BaseDataLoader):
    """测试不同的序列化方法"""
    print(f"\n🔬 序列化方法测试:")
    print(f"{'-'*60}")
    
    results = {}
    for method in methods:
        try:
            print(f"\n📋 测试方法: {method}")
            
            # 创建序列化器
            serializer = SerializerFactory.create_serializer(method)
            
            # 初始化序列化器（使用数据集加载器）
            serializer.initialize_with_dataset(dataloader, [sample])
            
            # 执行序列化
            result = serializer.serialize(sample)
            
            # 获取序列化结果
            token_sequences = result.token_sequences
            element_sequences = result.element_sequences
            
            print(f"   ✅ 序列化成功")
            print(f"   📏 序列数量: {len(token_sequences)}")
            
            # 显示第一个序列
            if token_sequences:
                token_seq = token_sequences[0]
                element_seq = element_sequences[0] if element_sequences else []
                
                print(f"   🔢 Token序列: {token_seq}")
                print(f"   📝 元素序列: {element_seq}")
                print(f"   📊 序列长度: {len(token_seq)}")
                
                # 尝试转换为可读字符串
                try:
                    readable_seq = serializer.tokens_to_string(token_seq)
                    print(f"   📖 可读序列: {readable_seq}")
                except Exception as e:
                    print(f"   ⚠️ 无法转换为可读字符串: {e}")
                
                # 保存结果
                results[method] = {
                    'token_sequences': token_sequences,
                    'element_sequences': element_sequences,
                    'length': len(token_seq),
                    'success': True
                }
            else:
                print(f"   ❌ 没有生成序列")
                results[method] = {'success': False, 'error': 'No sequences generated'}
        except Exception as e:
            import traceback
            print(f"   ❌ 序列化失败: {traceback.format_exc()}")
            results[method] = {'success': False, 'error': str(e)}
    
    return results

def compare_serialization_results(results: Dict[str, Dict[str, Any]]):
    """比较不同序列化方法的结果"""
    print(f"\n📊 序列化结果比较:")
    print(f"{'-'*60}")
    
    # 创建比较表格
    print(f"{'方法':<15} {'状态':<8} {'长度':<8} {'序列示例':<30}")
    print(f"{'-'*15} {'-'*8} {'-'*8} {'-'*30}")
    
    for method, result in results.items():
        if result.get('success', False):
            token_sequences = result.get('token_sequences', [])
            if token_sequences:
                token_seq = token_sequences[0]
                length = len(token_seq)
                # 显示序列的前10个token
                seq_preview = str(token_seq[:10]) + "..." if len(token_seq) > 10 else str(token_seq)
                print(f"{method:<15} {'✅':<8} {length:<8} {seq_preview:<30}")
            else:
                print(f"{method:<15} {'❌':<8} {'0':<8} {'无序列':<30}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{method:<15} {'❌':<8} {'-':<8} {error[:28]:<30}")

def visualize_molecular_structure(sample: Dict[str, Any]):
    """可视化分子结构（如果可能）"""
    print(f"\n🎨 分子结构可视化:")
    print(f"{'-'*60}")
    
    # 尝试使用RDKit可视化
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        smiles = sample.get('smiles', '')
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                print(f"✅ 成功解析SMILES: {smiles}")
                print(f"📊 分子式: {Chem.CalcMolFormula(mol)}")
                print(f"🔢 原子数: {mol.GetNumAtoms()}")
                print(f"🔗 键数: {mol.GetNumBonds()}")
                
                # 尝试生成2D坐标并保存图片
                try:
                    mol_2d = Chem.Mol(mol)
                    AllChem.Compute2DCoords(mol_2d)
                    
                    # 保存图片
                    img_filename = f"molecule_{sample.get('id', 'unknown')}.png"
                    img = Draw.MolToImage(mol_2d, size=(400, 300))
                    img.save(img_filename)
                    print(f"🖼️ 分子结构图已保存为: {img_filename}")
                    
                except Exception as e:
                    print(f"⚠️ 无法生成2D结构图: {e}")
            else:
                print(f"❌ 无法解析SMILES: {smiles}")
        else:
            print(f"⚠️ 没有SMILES信息")
            
    except ImportError:
        print(f"⚠️ RDKit未安装，无法可视化分子结构")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

def main():
    """主函数"""
    print("🚀 开始测试不同序列化方法...")
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    np.random.seed(42)
    
    # 配置
    config = ProjectConfig()
    config.subgraph_limt = 100  # 限制数据量
    
    # 可用的序列化方法
    serialization_methods = SerializerFactory.get_available_serializers()
    dataset='qm9test'
    
    # 通过 UDI 获取 DataLoader
    print(f"📂 加载{dataset}数据集...")
    udi = UnifiedDataInterface(config, dataset)
    loader = udi.get_dataset_loader()
    data,indices = loader.get_all_data_with_indices()  #
    
    if len(data) == 0:
        print("❌ 没有加载到数据")
        return False
    
    print(f"✅ 成功加载 {len(data)} 个{dataset}数据")
    
    # 随机选择3个分子进行测试
    num_samples = min(3, len(data))
    selected_indices = random.sample(range(len(data)), num_samples)
    selected_samples = [data[i] for i in selected_indices]
    
    print(f"🎯 随机选择了 {num_samples} 个数据进行测试")
    
    # 对每个选中的分子进行测试
    for i, sample in enumerate(selected_samples):
        # 打印分子信息
        print_molecule_info(sample, i)
        
        # 可视化分子结构
        # visualize_molecular_structure(sample)
        
        # 测试序列化方法
        results = test_serialization_methods(sample, serialization_methods,loader)
        
        # 比较结果
        compare_serialization_results(results)
        
        print(f"\n{'='*80}")
    
    print("🎉 所有测试完成！")
    return True
        

if __name__ == "__main__":
    main()
    # sys.exit(0 if success else 1) 