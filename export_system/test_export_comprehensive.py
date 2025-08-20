"""全面严格验证导出系统 - 验证所有图，所有数据集"""

import sys
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface
from export_system import create_true_exporter, load_data


def comprehensive_validate_dataset(dataset_name: str, config: ProjectConfig) -> bool:
    """全面验证单个数据集的所有图"""
    print(f"\n🔍 全面验证 {dataset_name} 数据集")
    print("=" * 70)
    
    try:
        # 1. 重新导出数据集
        print("📂 步骤1: 重新导出数据集...")
        exporter = create_true_exporter(dataset_name, config)
        exporter.export()
        
        # 2. 加载原始数据和导出数据
        print("📊 步骤2: 加载原始数据和导出数据...")
        
        # 加载原始数据
        udi = UnifiedDataInterface(config=config, dataset=dataset_name)
        udi.preload_graphs()
        loader = udi.get_dataset_loader()
        train_data, val_data, test_data, train_labels, val_labels, test_labels = loader.load_data()
        
        # 获取原始划分
        original_splits = udi.get_split_indices()
        
        # 合并原始数据
        all_data = train_data + val_data + test_data
        all_labels = train_labels + val_labels + test_labels
        original_graphs = [sample['dgl_graph'] for sample in all_data]
        
        # 加载导出数据
        output_file = Path("data/exported") / f"{dataset_name}_export.pkl"
        exported_data = load_data(output_file)
        
        total_graphs = len(original_graphs)
        print(f"  - 数据集规模: {total_graphs} 个图")
        print(f"  - 划分: 训练{len(original_splits['train'])}, 验证{len(original_splits['val'])}, 测试{len(original_splits['test'])}")
        
        # 3. 严格验证划分索引
        print("🔍 步骤3: 严格验证划分索引...")
        exported_splits = exported_data['splits']
        
        for split_name in ['train', 'val', 'test']:
            original_idx = np.array(original_splits[split_name], dtype=np.int64)
            exported_idx = exported_splits[split_name]
            
            if not np.array_equal(original_idx, exported_idx):
                print(f"❌ {split_name}划分索引不一致!")
                return False
            
            print(f"  ✅ {split_name}划分索引完全一致 (长度: {len(original_idx)})")
        
        # 4. 全面验证所有图的结构和特征
        print(f"🔍 步骤4: 全面验证所有 {total_graphs} 个图的结构和特征...")
        
        # 显示进度的间隔
        progress_interval = max(1, total_graphs // 20)  # 每5%显示一次进度
        
        for graph_idx in range(total_graphs):
            if graph_idx % progress_interval == 0:
                progress = (graph_idx / total_graphs) * 100
                print(f"    进度: {graph_idx}/{total_graphs} ({progress:.1f}%)")
            
            original_graph = original_graphs[graph_idx]
            exported_graph = exported_data['graphs'][graph_idx]
            
            # 4.1 基本结构验证
            if original_graph.num_nodes() != exported_graph['num_nodes']:
                print(f"❌ 图{graph_idx}: 节点数不匹配 {original_graph.num_nodes()} vs {exported_graph['num_nodes']}")
                return False
                
            if original_graph.num_edges() != len(exported_graph['src']):
                print(f"❌ 图{graph_idx}: 边数不匹配 {original_graph.num_edges()} vs {len(exported_graph['src'])}")
                return False
            
            # 4.2 边连接关系验证
            orig_src, orig_dst = original_graph.edges()
            orig_src_np = orig_src.numpy()
            orig_dst_np = orig_dst.numpy()
            exp_src = exported_graph['src']
            exp_dst = exported_graph['dst']
            
            if not np.array_equal(orig_src_np, exp_src):
                print(f"❌ 图{graph_idx}: 源节点连接不匹配")
                return False
                
            if not np.array_equal(orig_dst_np, exp_dst):
                print(f"❌ 图{graph_idx}: 目标节点连接不匹配")
                return False
            
            # 4.3 节点特征完全验证
            orig_num_nodes = original_graph.num_nodes()
            expected_node_tokens = loader.get_node_tokens_bulk(original_graph, list(range(orig_num_nodes)))
            expected_node_array = np.array(expected_node_tokens, dtype=np.int64)
            exported_node_feat = exported_graph['node_feat']
            
            if not np.array_equal(expected_node_array, exported_node_feat):
                print(f"❌ 图{graph_idx}: 节点特征不匹配")
                print(f"    期望形状: {expected_node_array.shape}, 导出形状: {exported_node_feat.shape}")
                return False
            
            # 4.4 边特征完全验证
            orig_num_edges = original_graph.num_edges()
            if orig_num_edges > 0:
                expected_edge_tokens = loader.get_edge_tokens_bulk(original_graph, list(range(orig_num_edges)))
                expected_edge_array = np.array(expected_edge_tokens, dtype=np.int64)
                exported_edge_feat = exported_graph['edge_feat']
                
                if not np.array_equal(expected_edge_array, exported_edge_feat):
                    print(f"❌ 图{graph_idx}: 边特征不匹配")
                    print(f"    期望形状: {expected_edge_array.shape}, 导出形状: {exported_edge_feat.shape}")
                    return False
        
        print(f"  ✅ 所有 {total_graphs} 个图的结构和特征验证通过")
        
        # 5. 验证所有标签一致性
        print("🔍 步骤5: 验证所有标签一致性...")
        exported_labels = exported_data['labels']
        
        if len(all_labels) != len(exported_labels):
            print(f"❌ 标签数量不匹配: {len(all_labels)} vs {len(exported_labels)}")
            return False
        
        label_mismatch_count = 0
        for graph_idx in range(len(all_labels)):
            original_label = all_labels[graph_idx]
            exported_label = exported_labels[graph_idx]
            
            if not _labels_equal(original_label, exported_label):
                label_mismatch_count += 1
                if label_mismatch_count <= 5:  # 只显示前5个不匹配的标签
                    print(f"❌ 图 {graph_idx} 标签不匹配: {original_label} vs {exported_label}")
                
        if label_mismatch_count > 0:
            print(f"❌ 总计 {label_mismatch_count} 个标签不匹配")
            return False
        
        print(f"  ✅ 所有 {len(all_labels)} 个标签验证通过")
        
        # 6. 验证图的边存储方式
        print("🔍 步骤6: 验证图的边存储方式...")
        sample_graph = original_graphs[0]
        sample_exported = exported_data['graphs'][0]
        
        orig_src, orig_dst = sample_graph.edges()
        edge_set = set(zip(orig_src.tolist(), orig_dst.tolist()))
        reverse_edge_set = set(zip(orig_dst.tolist(), orig_src.tolist()))
        is_undirected = len(edge_set & reverse_edge_set) > 0
        
        print(f"    - 图类型: {'无向图(双向边)' if is_undirected else '有向图'}")
        print(f"    - DGL边数: {len(orig_src)}")
        print(f"    - 导出边数: {len(sample_exported['src'])}")
        print("    ✅ 边存储方式验证通过")
        
        print(f"\n🎉 {dataset_name} 全面验证完全通过!")
        print(f"    验证了 {total_graphs} 个图的完整结构和特征")
        return True
        
    except Exception as e:
        print(f"❌ {dataset_name} 全面验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _labels_equal(label1, label2) -> bool:
    """比较两个标签是否相等"""
    try:
        if isinstance(label1, dict) and isinstance(label2, dict):
            if set(label1.keys()) != set(label2.keys()):
                return False
            for key in label1.keys():
                if abs(float(label1[key]) - float(label2[key])) > 1e-6:
                    return False
            return True
        elif isinstance(label1, (list, np.ndarray)) and isinstance(label2, (list, np.ndarray)):
            return np.allclose(np.array(label1), np.array(label2), atol=1e-6)
        else:
            return abs(float(label1) - float(label2)) < 1e-6
    except:
        return label1 == label2


def main():
    """全面验证主函数"""
    print("🎯 开始全面验证导出系统...")
    
    config = ProjectConfig()
    
    # 成功导出的所有数据集
    all_datasets = [
        "qm9", "zinc", "molhiv", "aqsol", "colors3", "proteins", 
        "dd", "mutagenicity", "coildel", "dblp", "twitter", 
        "synthetic", "peptides_func", "peptides_struct"
    ]
    
    success_count = 0
    total_graphs_validated = 0
    
    for dataset in all_datasets:
        if comprehensive_validate_dataset(dataset, config):
            success_count += 1
            
            # 统计验证的图数量
            try:
                output_file = Path("exported") / f"{dataset}_export.pkl"
                data = load_data(output_file)
                dataset_graphs = len(data['graphs'])
                total_graphs_validated += dataset_graphs
                print(f"✅ 进度: {success_count}/{len(all_datasets)} 数据集, 累计验证 {total_graphs_validated} 个图")
            except:
                print(f"✅ 进度: {success_count}/{len(all_datasets)} 数据集")
        else:
            print(f"❌ {dataset} 全面验证失败，停止测试")
            break
    
    print("\n" + "=" * 80)
    print(f"📊 全面验证结果:")
    print(f"    成功数据集: {success_count}/{len(all_datasets)}")
    print(f"    验证图总数: {total_graphs_validated}")
    
    if success_count == len(all_datasets):
        print("🎉 所有数据集全面验证完全通过!")
    else:
        print("💥 部分数据集验证失败!")


if __name__ == "__main__":
    main()