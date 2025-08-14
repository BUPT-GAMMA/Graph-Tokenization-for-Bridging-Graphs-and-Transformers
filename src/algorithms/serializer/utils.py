"""
图算法工具函数

提供图的创建、验证、分析等工具函数
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import random


class GraphUtils:
    """图算法工具类"""
    
    @staticmethod
    def create_random_graph(n: int, p: float = 0.5, weighted: bool = True, 
                          weight_range: Tuple[float, float] = (1, 10), 
                          seed: Optional[int] = None) -> np.ndarray:
        """创建随机图
        
        Args:
            n: 节点数量
            p: 边存在概率
            weighted: 是否加权
            weight_range: 权重范围
            seed: 随机种子
            
        Returns:
            邻接矩阵
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    weight = 1
                    if weighted:
                        weight = random.uniform(weight_range[0], weight_range[1])
                    adj_matrix[i][j] = weight
                    adj_matrix[j][i] = weight
                    
        return adj_matrix
        
    @staticmethod
    def create_complete_graph(n: int, weighted: bool = True,
                            weight_range: Tuple[float, float] = (1, 10),
                            seed: Optional[int] = None) -> np.ndarray:
        """创建完全图
        
        Args:
            n: 节点数量
            weighted: 是否加权
            weight_range: 权重范围
            seed: 随机种子
            
        Returns:
            邻接矩阵
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                weight = 1
                if weighted:
                    weight = random.uniform(weight_range[0], weight_range[1])
                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight
                
        return adj_matrix
        
    @staticmethod
    def create_cycle_graph(n: int, weighted: bool = True,
                          weight_range: Tuple[float, float] = (1, 10),
                          seed: Optional[int] = None) -> np.ndarray:
        """创建环图
        
        Args:
            n: 节点数量
            weighted: 是否加权
            weight_range: 权重范围
            seed: 随机种子
            
        Returns:
            邻接矩阵
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            j = (i + 1) % n
            weight = 1
            if weighted:
                weight = random.uniform(weight_range[0], weight_range[1])
            adj_matrix[i][j] = weight
            adj_matrix[j][i] = weight
            
        return adj_matrix
        
    @staticmethod
    def create_path_graph(n: int, weighted: bool = True,
                         weight_range: Tuple[float, float] = (1, 10),
                         seed: Optional[int] = None) -> np.ndarray:
        """创建路径图
        
        Args:
            n: 节点数量
            weighted: 是否加权
            weight_range: 权重范围  
            seed: 随机种子
            
        Returns:
            邻接矩阵
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        adj_matrix = np.zeros((n, n))
        
        for i in range(n - 1):
            j = i + 1
            weight = 1
            if weighted:
                weight = random.uniform(weight_range[0], weight_range[1])
            adj_matrix[i][j] = weight
            adj_matrix[j][i] = weight
            
        return adj_matrix
        
    @staticmethod
    def is_connected(adj_matrix: np.ndarray) -> bool:
        """检查图是否连通
        
        Args:
            adj_matrix: 邻接矩阵
            
        Returns:
            是否连通
        """
        n = adj_matrix.shape[0]
        if n <= 1:
            return True
            
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        
        while stack:
            current = stack.pop()
            for i in range(n):
                if adj_matrix[current][i] > 0 and not visited[i]:
                    visited[i] = True
                    stack.append(i)
                    count += 1
                    
        return count == n
        
    @staticmethod
    def has_hamilton_cycle_necessary_condition(adj_matrix: np.ndarray) -> bool:
        """检查哈密顿回路的必要条件
        
        Args:
            adj_matrix: 邻接矩阵
            
        Returns:
            是否满足必要条件
        """
        n = adj_matrix.shape[0]
        
        # 图必须连通
        if not GraphUtils.is_connected(adj_matrix):
            return False
            
        # Dirac定理：如果每个顶点的度数至少为n/2，则存在哈密顿回路
        degrees = np.sum(adj_matrix > 0, axis=1)
        if all(degree >= n // 2 for degree in degrees):
            return True
            
        # Ore定理：对于任意两个不相邻的顶点，它们的度数之和至少为n
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 0:  # 不相邻
                    if degrees[i] + degrees[j] < n:
                        return False
                        
        return True
        
    @staticmethod
    def calculate_path_weight(path: List[int], adj_matrix: np.ndarray) -> float:
        """计算路径权重
        
        Args:
            path: 路径
            adj_matrix: 邻接矩阵
            
        Returns:
            路径总权重
        """
        if not path or len(path) < 2:
            return 0.0
            
        total_weight = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if adj_matrix[u][v] == 0:
                return float('inf')  # 路径不存在
            total_weight += adj_matrix[u][v]
            
        return total_weight
        
    @staticmethod
    def is_valid_hamilton_path(path: List[int], adj_matrix: np.ndarray) -> bool:
        """验证是否为有效的哈密顿路径
        
        Args:
            path: 路径
            adj_matrix: 邻接矩阵
            
        Returns:
            是否为有效的哈密顿路径
        """
        n = adj_matrix.shape[0]
        
        # 检查路径长度
        if len(path) != n:
            return False
            
        # 检查是否访问了所有节点且无重复
        if len(set(path)) != n or max(path) >= n or min(path) < 0:
            return False
            
        # 检查边是否存在
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if adj_matrix[u][v] == 0:
                return False
                
        return True
        
    @staticmethod
    def is_valid_hamilton_cycle(cycle: List[int], adj_matrix: np.ndarray) -> bool:
        """验证是否为有效的哈密顿回路
        
        Args:
            cycle: 回路
            adj_matrix: 邻接矩阵
            
        Returns:
            是否为有效的哈密顿回路
        """
        if not cycle:
            return False
            
        # 如果最后一个节点与第一个节点相同，去掉最后一个
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
            
        # 验证为哈密顿路径
        if not GraphUtils.is_valid_hamilton_path(cycle, adj_matrix):
            return False
            
        # 检查最后一个节点到第一个节点的边
        if adj_matrix[cycle[-1]][cycle[0]] == 0:
            return False
            
        return True
        
    @staticmethod
    def graph_statistics(adj_matrix: np.ndarray) -> Dict:
        """计算图的统计信息
        
        Args:
            adj_matrix: 邻接矩阵
            
        Returns:
            统计信息字典
        """
        n = adj_matrix.shape[0]
        edges = np.sum(adj_matrix > 0) // 2  # 无向图
        degrees = np.sum(adj_matrix > 0, axis=1)
        
        stats = {
            'num_nodes': n,
            'num_edges': edges,
            'density': edges / (n * (n - 1) / 2) if n > 1 else 0,
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'is_connected': GraphUtils.is_connected(adj_matrix),
            'degree_sequence': sorted(degrees, reverse=True)
        }
        
        # 计算权重统计(如果是加权图)
        weights = adj_matrix[adj_matrix > 0]
        if len(weights) > 0 and not np.allclose(weights, 1.0):
            stats['weighted'] = True
            stats['avg_weight'] = np.mean(weights)
            stats['max_weight'] = np.max(weights)
            stats['min_weight'] = np.min(weights)
        else:
            stats['weighted'] = False
            
        return stats
        
    @staticmethod
    def visualize_graph(adj_matrix: np.ndarray, path: Optional[List[int]] = None,
                       title: str = "", figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None):
        """可视化图
        
        Args:
            adj_matrix: 邻接矩阵
            path: 要高亮显示的路径
            title: 图标题
            figsize: 图大小
            save_path: 保存路径
        """
        # 创建networkx图
        G = nx.from_numpy_array(adj_matrix)
        
        plt.figure(figsize=figsize)
        
        # 计算布局
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制所有边
        nx.draw_networkx_edges(G, pos, alpha=0.3, color='gray')
        
        # 如果有路径，高亮显示路径边
        if path:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            # 如果是回路，添加最后一条边
            if len(path) > 2 and adj_matrix[path[-1]][path[0]] > 0:
                path_edges.append((path[-1], path[0]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color='red', width=3, alpha=0.8)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.9)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 如果是加权图，显示边权重
        weights = adj_matrix[adj_matrix > 0]
        if len(weights) > 0 and not np.allclose(weights, 1.0):
            edge_labels = nx.get_edge_attributes(G, 'weight')
            # 格式化权重显示
            edge_labels = {k: f'{v:.1f}' for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            
        plt.title(title if title else "图可视化")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def compare_graphs(graphs: List[np.ndarray], labels: List[str] = None,
                      figsize: Tuple[int, int] = (15, 5)):
        """比较多个图
        
        Args:
            graphs: 图列表
            labels: 图标签列表
            figsize: 图大小
        """
        n_graphs = len(graphs)
        if labels is None:
            labels = [f'图 {i+1}' for i in range(n_graphs)]
            
        fig, axes = plt.subplots(1, n_graphs, figsize=figsize)
        if n_graphs == 1:
            axes = [axes]
            
        for i, (graph, label) in enumerate(zip(graphs, labels)):
            G = nx.from_numpy_array(graph)
            pos = nx.spring_layout(G, seed=42)
            
            nx.draw(G, pos, ax=axes[i], with_labels=True, 
                   node_color='lightblue', node_size=300,
                   font_size=8, font_weight='bold')
            axes[i].set_title(label)
            
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def generate_test_cases() -> Dict[str, np.ndarray]:
        """生成标准测试用例
        
        Returns:
            测试用例字典
        """
        test_cases = {}
        
        # 简单的有哈密顿回路的图
        test_cases['simple_cycle'] = GraphUtils.create_cycle_graph(4, weighted=False)
        
        # 完全图
        test_cases['complete_4'] = GraphUtils.create_complete_graph(4, weighted=False)
        
        # 路径图 (无哈密顿回路)  
        test_cases['path_4'] = GraphUtils.create_path_graph(4, weighted=False)
        
        # 复杂随机图
        test_cases['random_dense'] = GraphUtils.create_random_graph(6, p=0.7, weighted=True, seed=42)
        
        # 稀疏随机图
        test_cases['random_sparse'] = GraphUtils.create_random_graph(8, p=0.3, weighted=True, seed=42)
        
        # Petersen图的简化版本
        petersen = np.array([
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        ])
        test_cases['petersen_like'] = petersen
        
        return test_cases 