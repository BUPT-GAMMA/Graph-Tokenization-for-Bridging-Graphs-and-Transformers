#!/usr/bin/env python3
"""
简单导出功能测试
==============

测试数据集导出和转换功能是否正常。
"""

import os
from pathlib import Path
from simple_graph_loader import load_simple_graph_data, to_dgl, to_pyg, get_split_data


def test_export_and_load():
    """测试导出和加载功能"""
    
    # 测试文件列表
    test_files = [
        "qm9_simple.pkl",
        "zinc_simple.pkl", 
        "molhiv_simple.pkl"
    ]
    
    for file_name in test_files:
        if not os.path.exists(file_name):
            print(f"⚠️ 跳过 {file_name}（文件不存在）")
            continue
            
        print(f"\n🔍 测试 {file_name}")
        
        try:
            # 1. 加载数据
            data = load_simple_graph_data(file_name)
            print(f"✅ 数据加载成功: {len(data['graphs'])} 个图")
            
            # 2. 检查数据格式
            if data['graphs']:
                sample_graph = data['graphs'][0]
                print(f"   📊 样本图: {sample_graph['num_nodes']} 节点, {len(sample_graph['src'])} 条边")
                print(f"   🎯 节点特征维度: {len(sample_graph['node_feat'][0]) if sample_graph['node_feat'] else 0}")
                print(f"   🎯 边特征维度: {len(sample_graph['edge_feat'][0]) if sample_graph['edge_feat'] else 0}")
            
            # 3. 检查数据划分
            splits = data['splits']
            print(f"   📊 数据划分: {[(k, len(v)) for k, v in splits.items()]}")
            
            # 4. 测试获取训练集
            train_graphs, train_labels = get_split_data(data, 'train')
            print(f"   ✅ 训练集获取成功: {len(train_graphs)} 个图")
            
            # 5. 测试DGL转换（前5个图）
            try:
                test_data = {
                    'graphs': data['graphs'][:5],
                    'labels': data['labels'][:5]
                }
                dgl_data = to_dgl(test_data)
                print(f"   ✅ DGL转换成功: {len(dgl_data)} 个图")
                
                # 检查第一个图
                first_graph, first_label = dgl_data[0]
                print(f"      - DGL图: {first_graph.num_nodes()} 节点, {first_graph.num_edges()} 边")
                if 'feat' in first_graph.ndata:
                    print(f"      - 节点特征: {first_graph.ndata['feat'].shape}")
                if 'feat' in first_graph.edata:
                    print(f"      - 边特征: {first_graph.edata['feat'].shape}")
                    
            except ImportError:
                print("   ⚠️ 跳过DGL转换（DGL未安装）")
            except Exception as e:
                print(f"   ❌ DGL转换失败: {e}")
            
            # 6. 测试PyG转换（前5个图）
            try:
                test_data = {
                    'graphs': data['graphs'][:5],
                    'labels': data['labels'][:5]
                }
                pyg_data = to_pyg(test_data)
                print(f"   ✅ PyG转换成功: {len(pyg_data)} 个图")
                
                # 检查第一个图
                first_data = pyg_data[0]
                print(f"      - PyG图: {first_data.num_nodes} 节点, {first_data.edge_index.shape[1]} 边")
                print(f"      - 节点特征: {first_data.x.shape}")
                if first_data.edge_attr is not None:
                    print(f"      - 边特征: {first_data.edge_attr.shape}")
                print(f"      - 标签: {first_data.y}")
                
            except ImportError:
                print("   ⚠️ 跳过PyG转换（PyG未安装）")
            except Exception as e:
                print(f"   ❌ PyG转换失败: {e}")
                
        except Exception as e:
            print(f"❌ {file_name} 测试失败: {e}")
    
    print(f"\n🎉 测试完成！")


def run_exports():
    """运行所有导出脚本"""
    
    export_scripts = [
        "export_qm9.py",
        "export_zinc.py",
        "export_molhiv.py"
    ]
    
    print("🚀 开始运行导出脚本...")
    
    for script in export_scripts:
        if not os.path.exists(script):
            print(f"⚠️ 跳过 {script}（脚本不存在）")
            continue
            
        print(f"\n▶️ 运行 {script}")
        try:
            # 动态导入并执行
            module_name = script.replace('.py', '')
            exec(f"import {module_name}")
            
        except Exception as e:
            print(f"❌ {script} 执行失败: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试简单导出功能")
    parser.add_argument("--export", action="store_true", help="运行导出脚本")
    parser.add_argument("--test", action="store_true", help="测试已导出的数据")
    
    args = parser.parse_args()
    
    if args.export:
        run_exports()
    elif args.test:
        test_export_and_load()
    else:
        # 默认执行测试
        test_export_and_load()
