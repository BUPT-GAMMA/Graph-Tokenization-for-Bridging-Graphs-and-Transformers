"""真正正确的导出器 - 使用UDI和data loader标准接口获取token"""

import numpy as np
from typing import Dict, List, Any, Tuple

import dgl

from config import ProjectConfig
from src.data.unified_data_interface import UnifiedDataInterface


class TrueExporter:
    """真正正确的导出器 - 完全保持原有数据不变，仅获取token特征"""
    
    def __init__(self, dataset_name: str, config: ProjectConfig):
        self.dataset_name = dataset_name
        self.config = config
        
    def load_raw_data(self) -> Tuple[List[dgl.DGLGraph], List[Any], Dict[str, np.ndarray], Any]:
        """通过UDI接口原样加载数据"""
        udi = UnifiedDataInterface(config=self.config, dataset=self.dataset_name)
        udi.preload_graphs()
        loader = udi.get_dataset_loader()
        
        train_data, val_data, test_data, train_labels, val_labels, test_labels = loader.load_data()
        
        all_data = train_data + val_data + test_data
        all_labels = train_labels + val_labels + test_labels
        graphs = [sample['dgl_graph'] for sample in all_data]
        
        original_splits = udi.get_split_indices()
        splits = {
            'train': np.array(original_splits['train'], dtype=np.int64),
            'val': np.array(original_splits['val'], dtype=np.int64), 
            'test': np.array(original_splits['test'], dtype=np.int64)
        }
        
        return graphs, all_labels, splits, loader
    
    def convert_graph_to_export_format(self, graph: dgl.DGLGraph, loader) -> Dict[str, Any]:
        """使用data loader标准接口获取token特征"""
        src, dst = graph.edges()
        
        # 检查图必须有边
        num_edges = graph.num_edges()
        if num_edges == 0:
            raise ValueError(f"图不能没有边，节点数：{graph.num_nodes()}")
        
        return {
            'src': src.numpy().astype(np.int64),
            'dst': dst.numpy().astype(np.int64),
            'num_nodes': int(graph.num_nodes()),
            'node_feat': self._extract_node_tokens(graph, loader),
            'edge_feat': self._extract_edge_tokens(graph, loader)
        }
    
    def _extract_node_tokens(self, graph: dgl.DGLGraph, loader) -> np.ndarray:
        """使用data loader标准接口获取所有节点的token，转换为numpy数组"""
        num_nodes = graph.num_nodes()
        node_ids = list(range(num_nodes))
        node_tokens_list = loader.get_node_tokens_bulk(graph, node_ids)  # List[List[int]]
        return np.array(node_tokens_list, dtype=np.int64)
    
    def _extract_edge_tokens(self, graph: dgl.DGLGraph, loader) -> np.ndarray:
        """使用data loader标准接口获取所有边的token，转换为numpy数组"""
        num_edges = graph.num_edges()
        edge_ids = list(range(num_edges))
        edge_tokens_list = loader.get_edge_tokens_bulk(graph, edge_ids)  # List[List[int]]
        return np.array(edge_tokens_list, dtype=np.int64)

    def export(self):
        """执行导出"""
        graphs, labels, splits, loader = self.load_raw_data()
        
        exported_graphs = []
        for graph in graphs:
            exported_graphs.append(self.convert_graph_to_export_format(graph, loader))
        
        data = {
            'graphs': exported_graphs,
            'labels': labels,
            'splits': splits
        }
        
        import pickle
        from pathlib import Path
        
        output_file = Path("data/exported") / f"{self.dataset_name}_export.pkl" 
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)


def create_true_exporter(dataset_name: str, config: ProjectConfig) -> TrueExporter:
    """创建导出器"""
    return TrueExporter(dataset_name, config)