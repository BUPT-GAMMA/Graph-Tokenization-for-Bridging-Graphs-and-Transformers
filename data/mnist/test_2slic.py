import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from tensorflow.keras.datasets import mnist

# =============================================================================
# 函数 1: 创建自适应超像素图 (保持不变)
# =============================================================================
def create_adaptive_superpixel_graph(image, threshold=10, n_segments_digit=50, n_segments_bg=20):
    """
    输入一张图像，返回其自适应超像素图的相关数据。
    """
    digit_mask = image > threshold
    background_mask = ~digit_mask

    digit_segments = slic(image, n_segments=n_segments_digit, compactness=10, mask=digit_mask, start_label=1)
    background_segments = slic(image, n_segments=n_segments_bg, compactness=10, mask=background_mask, start_label=1)

    if background_segments.max() > 0:
        max_digit_label = digit_segments.max()
        background_segments[background_segments > 0] += max_digit_label

    final_segments = background_segments.copy()
    final_segments[digit_mask] = digit_segments[digit_mask]

    G = nx.Graph()
    node_positions = {}
    
    unique_labels = np.unique(final_segments)
    for segment_id in unique_labels:
        if segment_id == 0: continue
        coords = np.argwhere(final_segments == segment_id)
        if coords.size == 0: continue
        
        centroid = coords.mean(axis=0)
        rep_coord = min(coords, key=lambda c: (c[0]-centroid[0])**2 + (c[1]-centroid[1])**2)
        feature = int(image[rep_coord[0], rep_coord[1]])
        
        G.add_node(segment_id, feature=feature)
        node_positions[segment_id] = rep_coord

    rows, cols = image.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            current_seg = final_segments[r, c]
            right_seg = final_segments[r, c + 1]
            down_seg = final_segments[r + 1, c]
            
            if current_seg != right_seg and current_seg > 0 and right_seg > 0:
                G.add_edge(current_seg, right_seg)
            if current_seg != down_seg and current_seg > 0 and down_seg > 0:
                G.add_edge(current_seg, down_seg)
                
    return G, final_segments, node_positions

# =============================================================================
# 函数 2: 在指定的子图上进行可视化 (新封装)
# =============================================================================
def visualize_single_result(ax, image, graph, segments, positions, title=""):
    """
    在给定的 Matplotlib Axes 对象上绘制单个图像及其图的可视化结果。
    """
    # 绘制带边界的图像
    boundaries_on_image = mark_boundaries(image, segments, mode='subpixel')
    ax.imshow(boundaries_on_image)

    # 准备 networkx 绘图所需的坐标字典 (col, row)，并应用经验修正因子
    # 注意: *2 是根据用户反馈得到的经验修正，可能需要根据环境调整。
    pos_for_drawing = {
        node_id: (coords[1] * 2, coords[0] * 2) 
        for node_id, coords in positions.items()
    }

    # 绘制图的节点和边
    nx.draw_networkx(
        graph,
        pos=pos_for_drawing,
        ax=ax,
        with_labels=False,
        node_size=15,
        node_color='cyan',
        edge_color='yellow',
        width=1.0
    )

    ax.set_title(title, fontsize=10)
    ax.axis('off')

# =============================================================================
# 主执行流程
# =============================================================================
def main():
    # --- 1. 参数设置 ---
    NUM_SAMPLES = 100
    GRID_SIZE = 10  # 将会创建一个 10x10 的网格
    SAVE_PATH = "data/mnist/mnist_100_samples_visualization.png"

    # --- 2. 加载数据并随机选择样本 ---
    print("加载 MNIST 数据集...")
    (x_train, y_train), (_, _) = mnist.load_data()
    random_indices = np.random.choice(len(x_train), NUM_SAMPLES, replace=False)
    print(f"已随机选择 {NUM_SAMPLES} 个样本。")

    # --- 3. 创建可视化网格 ---
    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(22, 22))

    # --- 4. 循环处理并绘制每个样本 ---
    print("开始处理并绘制样本...")
    for i, ax in enumerate(axes.flat):
        # 获取当前样本
        img_idx = random_indices[i]
        image = x_train[img_idx]
        label = y_train[img_idx]
        
        print(f"  处理中: 样本 {i+1}/{NUM_SAMPLES} (图像索引: {img_idx}, 标签: {label})")

        # 执行图转换
        graph, segments, positions = create_adaptive_superpixel_graph(image)

        # 在对应的子图上进行可视化
        visualize_single_result(ax, image, graph, segments, positions, title=f"Label: {label}")
    
    # --- 5. 调整整体布局、保存并显示 ---
    print("\n所有样本处理完毕，正在生成最终图像...")
    fig.suptitle('MNIST 100个样本的自适应超像素图可视化', fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # 调整布局，为大标题留出空间

    # 确保保存目录存在
    save_dir = os.path.dirname(SAVE_PATH)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"网格图已保存到: {SAVE_PATH}")
    
    plt.show()

# --- 运行主函数 ---
if __name__ == '__main__':
    main()