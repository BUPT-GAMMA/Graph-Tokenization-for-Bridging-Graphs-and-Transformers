"""Graph serialization algorithms.
图序列化算法。"""

# Unified serializer interface
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
    # Core classes
    'BaseGraphSerializer',
    'SerializationResult',
    'GlobalIDMapping',
    
    # Serializer implementations
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

def create_serializer(method: str, **kwargs):
    """Create a serializer by method name."""
    return SerializerFactory.create_serializer(method, **kwargs) 