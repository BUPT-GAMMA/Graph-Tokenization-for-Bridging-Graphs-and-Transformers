"""统一数据加载器"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

import torch
import dgl

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYGEOMETRIC = True
except ImportError:
    PyGData = None
    HAS_PYGEOMETRIC = False


def load_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """加载导出的数据文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 检查所有图都有边
    for i, graph in enumerate(data['graphs']):
        if len(graph['src']) == 0:
            raise ValueError(f"图{i}没有边，这是不允许的")
    
    return data


def _create_dgl_graph(graph_dict: Dict[str, Any]) -> dgl.DGLGraph:
    """从图字典创建DGL图"""
    src = torch.from_numpy(graph_dict['src']).long()
    dst = torch.from_numpy(graph_dict['dst']).long()
    
    # 检查必须有边
    if len(src) == 0:
        raise ValueError("图没有边，无法创建DGL图")
    
    g = dgl.graph((src, dst), num_nodes=graph_dict['num_nodes'])
    
    # 处理节点特征：np.ndarray -> tensor
    node_feat = torch.from_numpy(graph_dict['node_feat']).long()
    g.ndata['feat'] = node_feat
    
    # 处理边特征：np.ndarray -> tensor  
    edge_feat = torch.from_numpy(graph_dict['edge_feat']).long()
    g.edata['feat'] = edge_feat
    
    return g


def _create_pyg_data(graph_dict: Dict[str, Any], label: Any) -> 'PyGData':
    """从图字典创建PyG Data对象"""
    src = graph_dict['src']
    dst = graph_dict['dst']
    
    # 检查必须有边
    if len(src) == 0:
        raise ValueError("图没有边，无法创建PyG图")
    
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    
    # 处理节点特征：np.ndarray -> tensor
    x = torch.from_numpy(graph_dict['node_feat']).long()
    
    # 处理边特征：np.ndarray -> tensor
    edge_attr = torch.from_numpy(graph_dict['edge_feat']).long()
    
    # 处理标签
    if isinstance(label, dict):
        y = torch.tensor(list(label.values())).float()
    elif isinstance(label, (list, np.ndarray)):
        y = torch.tensor(label)
    else:
        y = torch.tensor([label])
    
    return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=graph_dict['num_nodes'])


# 各数据集的转换函数
def to_dgl_qm9(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_qm9(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_zinc(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_zinc(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_molhiv(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_molhiv(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_aqsol(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_aqsol(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_colors3(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_colors3(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_proteins(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_proteins(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_dd(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_dd(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_mutagenicity(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_mutagenicity(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_code2(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_code2(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_coildel(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_coildel(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_dblp(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_dblp(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_twitter(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_twitter(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_synthetic(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_synthetic(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_mnist(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_mnist(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_peptides_func(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_peptides_func(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]
def to_dgl_peptides_struct(data): return [(_create_dgl_graph(g), label) for g, label in zip(data['graphs'], data['labels'])]
def to_pyg_peptides_struct(data): return [_create_pyg_data(g, label) for g, label in zip(data['graphs'], data['labels'])]