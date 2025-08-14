"""
图算法模块

包含各种图序列化算法的实现
"""

# 统一序列化器接口
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping
from .smiles_serializer import SMILESSerializer
from .dfs_serializer import DFSSerializer
from .bfs_serializer import BFSSerializer
from .eulerian_serializer import EulerianSerializer
from .chinese_postman_serializer import CPPSerializer
from .topo_serializer import TopoSerializer
from .freq_eulerian_serializer import FeulerSerializer
from .freq_chinese_postman_serializer import FCPPSerializer
from .serializer_factory import SerializerFactory

__all__ = [
    # 核心类
    'BaseGraphSerializer',
    'SerializationResult',
    'GlobalIDMapping',
    
    # 序列化器实现
    'SMILESSerializer',
    'DFSSerializer', 
    'BFSSerializer',
    'EulerianSerializer',
    'TopoSerializer',
    'FeulerSerializer',
    'CPPSerializer',
    'FCPPSerializer',
    'SerializerFactory',
]

# 推荐的序列化方式
def create_serializer(method: str, **kwargs):
    """
    推荐的序列化器创建方式 - 统一接口
    
    Args:
        method: 序列化方法名称
        **kwargs: 序列化器参数
        
    Returns:
        序列化器实例
        
    Example:
        >>> serializer = create_serializer('feuler', verbose=True, include_edge_tokens=False)
        >>> result = serializer.serialize(graph_data)
    """
    return SerializerFactory.create_serializer(method, **kwargs) 