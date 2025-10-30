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
        