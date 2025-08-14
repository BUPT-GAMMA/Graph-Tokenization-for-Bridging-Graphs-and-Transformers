"""
简洁高效的欧拉回路算法库
=======================

核心特性：
- 支持任意类型节点（泛型）
- 链式前向星高效图表示
- 三种核心算法：单一回路、变异生成、随机采样
- 智能起始节点检测
- 最小化代码量，最大化效率

Author: 基于原库优化简化
"""

import pandas as pd
import numpy as np
import random
from typing import List, Set, Tuple, Optional, Dict, Any, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

# 类型定义
NodeType = TypeVar('NodeType')

class GraphType(Enum):
    """图类型"""
    CIRCUIT = "circuit"
    MOLECULE = "molecule" 
    NETWORK = "network"
    GENERIC = "generic"

@dataclass
class Edge:
    """边结构 - 链式前向星"""
    to: int                 # 目标节点索引
    next_edge: int = -1     # 下一条边索引
    edge_id: int = 0        # 边ID

class EulerianGraph(Generic[NodeType]):
    """高效图表示 - 链式前向星 + 节点管理"""
    
    def __init__(self):
        # 节点管理
        self.node_to_idx: Dict[NodeType, int] = {}
        self.idx_to_node: Dict[int, NodeType] = {}
        self.node_count = 0
        
        # 链式前向星
        self.head: List[int] = []  # 每个节点的第一条边
        self.edges: List[Edge] = []  # 所有边
        self.edge_count = 0
    
    def add_node(self, node: NodeType) -> int:
        """添加节点，返回索引"""
        if node in self.node_to_idx:
            return self.node_to_idx[node]
        
        idx = self.node_count
        self.node_to_idx[node] = idx
        self.idx_to_node[idx] = node
        self.node_count += 1
        self.head.append(-1)
        return idx
    
    def add_edge(self, u: NodeType, v: NodeType):
        """添加无向边"""
        u_idx = self.add_node(u)
        v_idx = self.add_node(v)
        
        # 添加 u -> v
        edge_id = self.edge_count
        self.edges.append(Edge(v_idx, self.head[u_idx], edge_id))
        self.head[u_idx] = len(self.edges) - 1
        
        # 添加 v -> u (无向图)
        self.edges.append(Edge(u_idx, self.head[v_idx], edge_id))
        self.head[v_idx] = len(self.edges) - 1
        
        self.edge_count += 1
    
    def get_neighbors(self, node: NodeType) -> List[Tuple[NodeType, int]]:
        """获取邻居节点和边ID"""
        node_idx = self.node_to_idx.get(node)
        if node_idx is None:
            return []
        
        neighbors = []
        edge_idx = self.head[node_idx]
        while edge_idx != -1:
            edge = self.edges[edge_idx]
            neighbor = self.idx_to_node[edge.to]
            neighbors.append((neighbor, edge.edge_id))
            edge_idx = edge.next_edge
        return neighbors
    
    def get_all_nodes(self) -> List[NodeType]:
        """获取所有节点"""
        return list(self.node_to_idx.keys())

class EulerianFinder(Generic[NodeType]):
    """欧拉回路查找器"""
    
    def __init__(self, graph: EulerianGraph[NodeType], graph_type: GraphType = GraphType.GENERIC):
        self.graph = graph
        self.graph_type = graph_type
    
    def detect_start_node(self, preference: Optional[NodeType] = None) -> NodeType:
        """智能检测起始节点"""
        if preference and preference in self.graph.node_to_idx:
            return preference
        
        nodes = self.graph.get_all_nodes()
        if not nodes:
            raise ValueError("图为空")
        
        if self.graph_type == GraphType.CIRCUIT:
            # 电路图：优先选择电源节点
            power_keywords = ['VSS', 'VDD', 'GND', 'VCC']
            for node in nodes:
                if any(keyword in str(node).upper() for keyword in power_keywords):
                    return node
        
        # 默认：选择度数最小的节点
        min_node = min(nodes, key=lambda n: len(self.graph.get_neighbors(n)))
        return min_node
    
    def find_single_circuit(self, start_node: Optional[NodeType] = None) -> Optional[List[NodeType]]:
        """查找单条欧拉回路 - 最高效"""
        if start_node is None:
            start_node = self.detect_start_node()
        
        visited_edges = set()
        path = []
        
        def dfs(node: NodeType):
            path.append(node)
            for neighbor, edge_id in self.graph.get_neighbors(node):
                if edge_id not in visited_edges:
                    visited_edges.add(edge_id)
                    dfs(neighbor)
                    path.append(node)
        
        dfs(start_node)
        
        # 验证是否覆盖所有边
        if len(visited_edges) == self.graph.edge_count:
            return path
        return None
    
    def find_multiple_by_mutation(self, 
                                 start_node: Optional[NodeType] = None,
                                 max_solutions: int = 50,
                                 mutation_rounds: int = 5) -> List[List[NodeType]]:
        """通过变异生成多条回路"""
        if start_node is None:
            start_node = self.detect_start_node()
        
        # 获取初始路径
        initial = self.find_single_circuit(start_node)
        if not initial:
            return []
        
        all_paths = [initial]
        unique_paths = {tuple(str(n) for n in initial)}
        
        for _ in range(mutation_rounds):
            if len(all_paths) >= max_solutions:
                break
            
            current_paths = all_paths.copy()
            for base_path in current_paths:
                if len(all_paths) >= max_solutions:
                    break
                
                # 在随机位置变异
                for i in range(0, len(base_path)-1, max(1, len(base_path)//10)):
                    mutated = self._mutate_path(base_path, i)
                    if mutated:
                        path_key = tuple(str(n) for n in mutated)
                        if path_key not in unique_paths:
                            unique_paths.add(path_key)
                            all_paths.append(mutated)
                            if len(all_paths) >= max_solutions:
                                break
        
        return all_paths
    
    def find_multiple_by_sampling(self,
                                 start_node: Optional[NodeType] = None,
                                 max_solutions: int = 50,
                                 max_attempts: int = 200,
                                 seed: Optional[int] = None) -> List[List[NodeType]]:
        """通过随机采样生成多条回路"""
        if start_node is None:
            start_node = self.detect_start_node()
        
        if seed is not None:
            random.seed(seed)
        
        all_paths = []
        unique_paths = set()
        
        for _ in range(max_attempts):
            if len(all_paths) >= max_solutions:
                break
            
            path = self._random_dfs(start_node)
            if path:
                path_key = tuple(str(n) for n in path)
                if path_key not in unique_paths:
                    unique_paths.add(path_key)
                    all_paths.append(path)
        
        return all_paths
    
    def _mutate_path(self, path: List[NodeType], pos: int) -> Optional[List[NodeType]]:
        """在指定位置变异路径"""
        if pos >= len(path) - 1:
            return None
        
        # 记录已使用的边
        used_edges = set()
        for i in range(pos):
            if i < len(path) - 1:
                edge_id = self._get_edge_id(path[i], path[i+1])
                if edge_id is not None:
                    used_edges.add(edge_id)
        
        # 寻找新的邻居
        current = path[pos]
        for neighbor, edge_id in self.graph.get_neighbors(current):
            if edge_id not in used_edges:
                # 构建新路径
                new_path = path[:pos+1]
                extended = self._dfs_from(neighbor, used_edges.copy(), new_path.copy())
                if self._is_valid_circuit(extended):
                    return extended
        return None
    
    def _random_dfs(self, start_node: NodeType) -> Optional[List[NodeType]]:
        """随机化DFS"""
        visited_edges = set()
        path = []
        
        def dfs(node: NodeType):
            path.append(node)
            neighbors = self.graph.get_neighbors(node)
            random.shuffle(neighbors)
            
            for neighbor, edge_id in neighbors:
                if edge_id not in visited_edges:
                    visited_edges.add(edge_id)
                    dfs(neighbor)
                    path.append(node)
        
        dfs(start_node)
        return path if len(visited_edges) == self.graph.edge_count else None
    
    def _dfs_from(self, start: NodeType, used_edges: Set[int], path: List[NodeType]) -> List[NodeType]:
        """从指定节点继续DFS"""
        def dfs(node: NodeType):
            path.append(node)
            for neighbor, edge_id in self.graph.get_neighbors(node):
                if edge_id not in used_edges:
                    used_edges.add(edge_id)
                    dfs(neighbor)
                    path.append(node)
        
        dfs(start)
        return path
    
    def _get_edge_id(self, u: NodeType, v: NodeType) -> Optional[int]:
        """获取边ID"""
        for neighbor, edge_id in self.graph.get_neighbors(u):
            if neighbor == v:
                return edge_id
        return None
    
    def _is_valid_circuit(self, path: List[NodeType]) -> bool:
        """验证是否为有效欧拉回路"""
        if len(path) < 2:
            return False
        
        used_edges = set()
        for i in range(len(path) - 1):
            edge_id = self._get_edge_id(path[i], path[i+1])
            if edge_id is None or edge_id in used_edges:
                return False
            used_edges.add(edge_id)
        
        return len(used_edges) == self.graph.edge_count

# 便捷构建函数
def from_adjacency_matrix(matrix: pd.DataFrame, graph_type: GraphType = GraphType.GENERIC) -> EulerianFinder[str]:
    """从邻接矩阵构建"""
    graph = EulerianGraph[str]()
    
    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i, j] == 1:
                graph.add_edge(str(i), str(j))
    
    return EulerianFinder(graph, graph_type)

def from_edge_list(edges: List[Tuple[NodeType, NodeType]], 
                  graph_type: GraphType = GraphType.GENERIC) -> EulerianFinder[NodeType]:
    """从边列表构建"""
    graph = EulerianGraph[NodeType]()
    
    for u, v in edges:
        graph.add_edge(u, v)
    
    return EulerianFinder(graph, graph_type)

def from_circuit_netlist(netlist: Dict[str, List[str]]) -> EulerianFinder[str]:
    """从电路网表构建"""
    graph = EulerianGraph[str]()
    
    for component, pins in netlist.items():
        for i in range(len(pins)):
            for j in range(i + 1, len(pins)):
                graph.add_edge(pins[i], pins[j])
    
    return EulerianFinder(graph, GraphType.CIRCUIT)

# 批量处理函数
def process_dataset(base_dir: str, start_idx: int = 1, end_idx: int = 100, 
                   save_results: bool = True) -> None:
    """批量处理数据集"""
    import os
    
    for i in range(start_idx, end_idx + 1):
        csv_file = f"{base_dir}/{i}/Graph{i}.csv"
        if not os.path.exists(csv_file):
            continue
        
        try:
            matrix = pd.read_csv(csv_file, index_col=0)
            finder = from_adjacency_matrix(matrix, GraphType.CIRCUIT)
            
            # 根据图大小选择算法
            max_solutions = 20 if i > 1280 else 100
            circuits = finder.find_multiple_by_mutation(max_solutions=max_solutions)
            
            if circuits:
                # 填充到固定长度
                padded = [path + ['TRUNCATE'] * (1025 - len(path)) 
                         for path in circuits if len(path) <= 1025]
                
                print(f"处理 {base_dir}/{i}: {len(padded)} 条回路")
                
                if save_results:
                    save_file = f"{base_dir}/{i}/Sequence_total{i}.npy"
                    np.save(save_file, padded)
            
        except Exception as e:
            print(f"处理 {base_dir}/{i} 出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 电路图
    print("=== 电路图示例 ===")
    try:
        matrix = pd.read_csv("Dataset/1/Graph1.csv", index_col=0)
        finder = from_adjacency_matrix(matrix, GraphType.CIRCUIT)
        
        # 单一回路
        circuit = finder.find_single_circuit()
        print(f"找到回路: {len(circuit) if circuit else 0} 个节点")
        
        # 多回路生成
        circuits = finder.find_multiple_by_mutation(max_solutions=10)
        print(f"变异生成: {len(circuits)} 条回路")
        
        circuits = finder.find_multiple_by_sampling(max_solutions=10)
        print(f"采样生成: {len(circuits)} 条回路")
        
    except FileNotFoundError:
        print("文件不存在，使用示例数据")
        
        # 简单示例
        edges = [("VSS", "A"), ("A", "B"), ("B", "VDD"), ("VDD", "VSS")]
        finder = from_edge_list(edges, GraphType.CIRCUIT)
        circuit = finder.find_single_circuit()
        print(f"示例回路: {circuit}")
    
    # 示例2: 分子图
    print("\n=== 分子图示例 ===")
    # 甲烷分子
    molecule_edges = [("C", "H1"), ("C", "H2"), ("C", "H3"), ("C", "H4")]
    mol_finder = from_edge_list(molecule_edges, GraphType.MOLECULE)
    mol_circuit = mol_finder.find_single_circuit()
    print(f"分子回路: {mol_circuit}")
    
    # 示例3: 网络图
    print("\n=== 网络图示例 ===")
    network_edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
    net_finder = from_edge_list(network_edges, GraphType.NETWORK)
    net_circuits = net_finder.find_multiple_by_sampling(max_solutions=5, seed=42)
    print(f"网络回路: {len(net_circuits)} 条")
    
    # 批量处理示例（注释掉避免实际运行）
    # process_dataset("Dataset", 1, 10) 