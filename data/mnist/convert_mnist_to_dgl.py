import os
import json
import pickle
import numpy as np
import networkx as nx
import dgl
import torch
from tensorflow.keras.datasets import mnist
from final_slic import process_mnist_to_graph, GRAPH_PARAMS
from sklearn.model_selection import train_test_split
import time

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def convert_nx_to_dgl(nx_graph:nx.Graph, label:int):
    """
    将NetworkX图转换为DGL图格式
    
    参数:
    - nx_graph: NetworkX图对象
    - label: 图的标签（数字0-9）
    
    返回:
    - dgl_graph: DGL图对象
    """
    # 将无向图转换为有向图，这样DGL可以正确处理边特征
    nx_directed = nx_graph.to_directed()
    
    # 创建DGL图，包含节点和边特征
    dgl_graph = dgl.from_networkx(nx_directed, node_attrs=['feature'], edge_attrs=['feature'])
    
    return dgl_graph

def process_mnist_dataset():
    """
    处理MNIST数据集，转换为DGL图格式
    
    参数:
    - num_samples: 处理的样本数量，None表示处理全部
    - random_seed: 随机种子
    
    返回:
    - graph_label_pairs: (graph, label)元组列表
    - indices: 原始索引列表
    """
    print("加载MNIST数据集...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 合并训练集和测试集
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    
    _,x_all,_,y_all=train_test_split(x_all,y_all,test_size=0.1,random_state=seed)
    
    print(f"开始处理 {len(x_all)} 张图像...")
    
    graph_label_pairs = []
    failed_conversions = 0
    
    start_time = time.time()
    
    for i, (image, label) in enumerate(zip(x_all, y_all)):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(x_all)} ({(i/len(x_all)*100):.1f}%)")
        
        try:
            # 使用process_mnist_to_graph函数转换图像为图
            nx_graph, segments = process_mnist_to_graph(image, GRAPH_PARAMS)
            
            # 检查图是否有效（至少有一个节点）
            if nx_graph.number_of_nodes() > 0:
                # 转换为DGL格式
                dgl_graph = convert_nx_to_dgl(nx_graph, label)
                # 将图和标签作为元组存储
                graph_label_pairs.append((dgl_graph, label))
            else:
                failed_conversions += 1
                print(f"警告: 图像 {i} 转换后没有节点，跳过")
                
        except Exception as e:
            failed_conversions += 1
            print(f"错误: 处理图像 {i} 时出错: {e}")
            continue
    
    end_time = time.time()
    print(f"处理完成！")
    print(f"成功转换: {len(graph_label_pairs)} 个图")
    print(f"转换失败: {failed_conversions} 个")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    
    return graph_label_pairs

def create_train_test_val_split(graph_label_pairs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    创建训练/验证/测试集划分
    
    参数:
    - graph_label_pairs: (graph, label)元组列表
    - indices: 原始索引列表
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例
    - test_ratio: 测试集比例
    
    返回:
    - train_indices, val_indices, test_indices: 各集合的索引
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 提取标签用于分层采样
    labels = [label for _, label in graph_label_pairs]
    
    # 首先分割出测试集
    train_val_indices, test_indices = train_test_split(
        np.arange(len(graph_label_pairs)), 
        test_size=test_ratio, 
        random_state=seed,
        stratify=labels
    )
    
    # 从剩余数据中分割训练集和验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=[labels[i] for i in train_val_indices]
    )
    
    return train_indices, val_indices, test_indices

def save_data(graph_label_pairs, train_indices, val_indices, test_indices, output_dir):
    """
    保存数据到指定目录
    
    参数:
    - graph_label_pairs: (graph, label)元组列表
    - train_indices, val_indices, test_indices: 各集合的索引
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存DGL图数据
    print("保存DGL图数据...")
    # for graph, label in graph_label_pairs:
    #     # 将tensor特征转换回Python列表，避免存储tensor
    #     if graph.num_nodes() > 0:
    #         # 转换节点特征
    #         node_features = graph.ndata['feature'].tolist()
    #         graph.ndata['feature'] = node_features
    
    #     if graph.num_edges() > 0:
    #         # 转换边特征
    #         edge_features = graph.edata['feature'].tolist()
    #         graph.edata['feature'] = edge_features
    
    data_path = os.path.join(output_dir, "data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(graph_label_pairs, f)
    
    # 保存索引文件
    print("保存索引文件...")
    
    # 将numpy数组转换为Python列表以便JSON序列化
    train_indices_list = train_indices.tolist()
    val_indices_list = val_indices.tolist()
    test_indices_list = test_indices.tolist()
    
    with open(os.path.join(output_dir, "train_index.json"), 'w') as f:
        json.dump(train_indices_list, f)
    
    with open(os.path.join(output_dir, "val_index.json"), 'w') as f:
        json.dump(val_indices_list, f)
    
    with open(os.path.join(output_dir, "test_index.json"), 'w') as f:
        json.dump(test_indices_list, f)
    
    # 提取标签用于统计
    labels = [label for _, label in graph_label_pairs]
    
    # 保存转换统计信息
    stats = {
        "train": {
            "graphs": len(train_indices),
            "labels_distribution": {i: labels.count(i) for i in range(10)}
        },
        "val": {
            "graphs": len(val_indices),
            "labels_distribution": {i: labels.count(i) for i in range(10)}
        },
        "test": {
            "graphs": len(test_indices),
            "labels_distribution": {i: labels.count(i) for i in range(10)}
        },
        "total": {
            "graphs": len(graph_label_pairs),
        }
    }
    
    with open(os.path.join(output_dir, "conversion_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"数据已保存到: {output_dir}")
    print(f"训练集: {len(train_indices)} 个图")
    print(f"验证集: {len(val_indices)} 个图")
    print(f"测试集: {len(test_indices)} 个图")

def main():
    """主函数"""
    # 配置参数
    OUTPUT_DIR = "."
    
    print("=" * 50)
    print("MNIST数据集转换为DGL图格式")
    print("=" * 50)
    
    # 处理数据集（处理全部数据）
    graph_label_pairs = process_mnist_dataset()
    
    if len(graph_label_pairs) == 0:
        print("错误: 没有成功转换任何图")
        return
    
    # 创建数据集划分
    print("\n创建数据集划分...")
    train_indices, val_indices, test_indices = create_train_test_val_split(
        graph_label_pairs
    )
    
    # 保存数据
    print("\n保存数据...")
    save_data(graph_label_pairs, train_indices, val_indices, test_indices, OUTPUT_DIR)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("转换完成！统计信息:")
    print("=" * 50)
    
    # 提取标签和图的统计信息
    labels = [label for _, label in graph_label_pairs]
    graphs = [graph for graph, _ in graph_label_pairs]
    
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("标签分布:")
    for i in range(10):
        count = label_counts.get(i, 0)
        print(f"  数字 {i}: {count} 个")
    
    print(f"\n图统计:")
    print(f"  平均节点数: {np.mean([g.num_nodes() for g in graphs]):.1f}")
    print(f"  平均边数: {np.mean([g.num_edges() for g in graphs]):.1f}")
    print(f"  最大节点数: {max([g.num_nodes() for g in graphs])}")
    print(f"  最大边数: {max([g.num_edges() for g in graphs])}")

if __name__ == "__main__":
    main() 