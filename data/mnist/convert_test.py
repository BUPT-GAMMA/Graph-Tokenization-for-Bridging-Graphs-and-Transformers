import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from tensorflow.keras.datasets import mnist

# =============================================================================
# 1. 数据准备：加载一张 MNIST 图像
# =============================================================================
(x_train, y_train), (_, _) = mnist.load_data()
# 让我们选择一张手写数字 '8' 的图像作为示例
image_index = 10 
image = x_train[image_index]
label = y_train[image_index]

print(f"成功加载一张 MNIST 图像，数字为: {label}")
print(f"图像尺寸: {image.shape}")


# =============================================================================
# 2. 核心函数：将图像转换为超像素图
# =============================================================================
def create_superpixel_graph(image, n_segments=75, compactness=10):
    """
    将单张图像转换为一个保留离散特征的超像素图。

    参数:
    - image (np.array): 输入的2D灰度图像。
    - n_segments (int): 目标超像素（即图节点）的数量。
    - compactness (float): SLIC算法的紧凑度参数。

    返回:
    - G (nx.Graph): 构建好的 NetworkX 图。
    - segments (np.array): 超像素的标签掩码。
    - node_positions (dict): 节点ID到其代表性像素坐标的映射。
    """
    # --- 步骤 1: 超像素分割 ---
    # SLIC算法需要浮点数图像，并且最好是多通道的，我们通过堆叠来模拟
    image_float = image.astype(float)
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, 
                    sigma=1, start_label=0,max_size_factor=1000,min_size_factor=0.01)
    
    # 获取图中实际的节点数
    num_actual_nodes = len(np.unique(segments))
    
    # --- 步骤 2: 创建节点并赋予离散特征 ---
    G = nx.Graph()
    node_positions = {}
    
    for segment_id in range(num_actual_nodes):
        # 找到属于当前超像素的所有像素坐标 (row, col)
        coords = np.argwhere(segments == segment_id)
        
        # 计算质心
        centroid = coords.mean(axis=0)
        
        # 找到离质心最近的真实像素作为代表
        # key=lambda c: ... 计算每个坐标c到质心的欧氏距离的平方
        representative_coord = min(coords, key=lambda c: (c[0]-centroid[0])**2 + (c[1]-centroid[1])**2)
        
        # 提取该代表性像素的原始离散值作为特征
        discrete_feature = int(image[representative_coord[0], representative_coord[1]])
        
        # 将节点添加到图中，并存储其属性
        G.add_node(
            segment_id,
            feature=discrete_feature,
            pos=(representative_coord[1], representative_coord[0]) # (x, y) for plotting
        )
        
        # 存储节点位置用于后续绘图
        node_positions[segment_id] = representative_coord

    # --- 步骤 3: 构建边 ---
    # 遍历像素，检查相邻像素的ID是否不同
    rows, cols = image.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            current_seg = segments[r, c]
            # 检查右边和下边的邻居
            right_seg = segments[r, c + 1]
            down_seg = segments[r + 1, c]
            
            if current_seg != right_seg:
                G.add_edge(current_seg, right_seg)
            if current_seg != down_seg:
                G.add_edge(current_seg, down_seg)
                
    return G, segments, node_positions


# =============================================================================
# 3. 执行转换和可视化
# =============================================================================
# 调用函数进行转换
# 可以调整 n_segments 来改变图的规模
graph, superpixel_mask, positions = create_superpixel_graph(image, n_segments=100,compactness=0.1)

print("\n图转换完成:")
print(f" - 节点数: {graph.number_of_nodes()}")
print(f" - 边数: {graph.number_of_edges()}")

# 从图中取一个节点的特征查看
sample_node_id = list(graph.nodes)[0]
print(f" - 示例节点 {sample_node_id} 的特征 (离散像素值): {graph.nodes[sample_node_id]['feature']}")


# --- 可视化 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 图1: 超像素分割结果
# mark_boundaries 会在原图上用彩色线条描绘出超像素的边界
ax1 = axes[0]
ax1.imshow(mark_boundaries(image, superpixel_mask,mode='subpixel'))
ax1.set_title("1. 超像素分割结果 (Superpixel Segmentation)")
ax1.axis('off')

# 图2: 最终生成的图结构
ax2 = axes[1]
# 先把原始图像作为背景
ax2.imshow(image, cmap='gray')
# 提取节点位置用于绘图 (networkx 需要 (x,y) 格式)
pos_for_drawing = {node: (coords[1], coords[0]) for node, coords in positions.items()}

# 绘制节点
nx.draw_networkx_nodes(
    graph,
    pos=pos_for_drawing,
    ax=ax2,
    node_size=50,
    node_color='cyan'
)
# 绘制边
nx.draw_networkx_edges(
    graph,
    pos=pos_for_drawing,
    ax=ax2,
    edge_color='yellow',
    width=1.5
)
ax2.set_title("2. 最终生成的图 (Resulting Graph)")
ax2.axis('off')

plt.tight_layout()
plt.savefig("data/mnist/mnist_superpixel_graph.png")