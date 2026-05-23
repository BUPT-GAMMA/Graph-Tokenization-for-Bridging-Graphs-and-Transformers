"""Graph algorithm utilities.
图算法工具集。"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import random


class GraphUtils:
    """Graph algorithm utility class."""
        
    @staticmethod
    def is_connected(adj_matrix: np.ndarray) -> bool:
        """Check if graph is connected."""
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
        """Check necessary conditions for Hamiltonian cycle."""
        n = adj_matrix.shape[0]
        
        # Graph must be connected
        if not GraphUtils.is_connected(adj_matrix):
            return False
            
        # Dirac's theorem: if every vertex has degree >= n/2, Hamiltonian cycle exists
        degrees = np.sum(adj_matrix > 0, axis=1)
        if all(degree >= n // 2 for degree in degrees):
            return True
            
        # Ore's theorem: for any two non-adjacent vertices, degree sum >= n
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j] == 0:
                    if degrees[i] + degrees[j] < n:
                        return False
                        
        return True
        
    @staticmethod
    def calculate_path_weight(path: List[int], adj_matrix: np.ndarray) -> float:
        """Compute total path weight."""
        if not path or len(path) < 2:
            return 0.0
            
        total_weight = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if adj_matrix[u][v] == 0:
                return float('inf')  # edge does not exist
            total_weight += adj_matrix[u][v]
            
        return total_weight
        
    @staticmethod
    def is_valid_hamilton_path(path: List[int], adj_matrix: np.ndarray) -> bool:
        """Validate Hamiltonian path."""
        n = adj_matrix.shape[0]
        
        # Check path length
        if len(path) != n:
            return False
            
        # All nodes visited exactly once
        if len(set(path)) != n or max(path) >= n or min(path) < 0:
            return False
            
        # Check edges exist
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if adj_matrix[u][v] == 0:
                return False
                
        return True
        
    @staticmethod
    def is_valid_hamilton_cycle(cycle: List[int], adj_matrix: np.ndarray) -> bool:
        """Validate Hamiltonian cycle."""
        if not cycle:
            return False
            
        # Strip trailing duplicate of first node
        if len(cycle) > 1 and cycle[0] == cycle[-1]:
            cycle = cycle[:-1]
            
        # Validate as Hamiltonian path
        if not GraphUtils.is_valid_hamilton_path(cycle, adj_matrix):
            return False
            
        # Check closing edge
        if adj_matrix[cycle[-1]][cycle[0]] == 0:
            return False
            
        return True
        
    @staticmethod
    def graph_statistics(adj_matrix: np.ndarray) -> Dict:
        """Compute graph statistics."""
        n = adj_matrix.shape[0]
        edges = np.sum(adj_matrix > 0) // 2  # undirected
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
        
        # Weight stats (if weighted)
        weights = adj_matrix[adj_matrix > 0]
        if len(weights) > 0 and not np.allclose(weights, 1.0):
            stats['weighted'] = True
            stats['avg_weight'] = np.mean(weights)
            stats['max_weight'] = np.max(weights)
            stats['min_weight'] = np.min(weights)
        else:
            stats['weighted'] = False
            
        return stats
        