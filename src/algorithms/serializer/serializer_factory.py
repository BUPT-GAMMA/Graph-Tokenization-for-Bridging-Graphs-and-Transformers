"""
序列化器工厂
"""

from typing import Dict, Any
from .base_serializer import BaseGraphSerializer
from .smiles_serializer import SMILESSerializer
from .dfs_serializer import DFSSerializer
from .bfs_serializer import BFSSerializer
from .eulerian_serializer import EulerianSerializer
from .chinese_postman_serializer import CPPSerializer
from .topo_serializer import TopoSerializer
from .freq_eulerian_serializer import FeulerSerializer
from .freq_chinese_postman_serializer import FCPPSerializer
from .image_row_major_serializer import ImageRowMajorSerializer
from .image_serpentine_serializer import ImageSerpentineSerializer
from .image_diag_zigzag_serializer import ImageDiagZigzagSerializer

class SerializerFactory:
    """序列化器工厂类"""
    
    @staticmethod
    def create_serializer(serializer_type: str, **kwargs) -> BaseGraphSerializer:
        """
        创建序列化器
        
        Args:
            serializer_type: 序列化器类型
            **kwargs: 序列化器参数
            
        Returns:
            BaseGraphSerializer: 序列化器实例
        """
        if serializer_type == "smiles":
            return SMILESSerializer()
        elif serializer_type == "smiles_1":
            return SMILESSerializer(smiles_type="smiles_1")
        elif serializer_type == "smiles_2":
            return SMILESSerializer(smiles_type="smiles_2")
        elif serializer_type == "smiles_3":
            return SMILESSerializer(smiles_type="smiles_3")
        elif serializer_type == "smiles_4":
            return SMILESSerializer(smiles_type="smiles_4")
        elif serializer_type == "dfs":
            return DFSSerializer()
        elif serializer_type == "bfs":
            return BFSSerializer()
        elif serializer_type == "eulerian":
            kwargs_to_pass = {}
            if 'include_edge_tokens' in kwargs:
                kwargs_to_pass['include_edge_tokens'] = kwargs['include_edge_tokens']
            if 'omit_most_frequent_edge' in kwargs:
                kwargs_to_pass['omit_most_frequent_edge'] = kwargs['omit_most_frequent_edge']
            return EulerianSerializer(**kwargs_to_pass)
        elif serializer_type == "cpp":
            kwargs_to_pass = {}
            if 'include_edge_tokens' in kwargs:
                kwargs_to_pass['include_edge_tokens'] = kwargs['include_edge_tokens']
            if 'omit_most_frequent_edge' in kwargs:
                kwargs_to_pass['omit_most_frequent_edge'] = kwargs['omit_most_frequent_edge']
            if 'verbose' in kwargs:
                kwargs_to_pass['verbose'] = kwargs['verbose']
            return CPPSerializer(**kwargs_to_pass)
        elif serializer_type == "topo":
            return TopoSerializer()
        elif serializer_type == "feuler":
            verbose = kwargs.get('verbose', False)
            kwargs_to_pass = {}
            if 'include_edge_tokens' in kwargs:
                kwargs_to_pass['include_edge_tokens'] = kwargs['include_edge_tokens']
            if 'omit_most_frequent_edge' in kwargs:
                kwargs_to_pass['omit_most_frequent_edge'] = kwargs['omit_most_frequent_edge']
            return FeulerSerializer(verbose=verbose, **kwargs_to_pass)
        elif serializer_type == "fcpp":
            verbose = kwargs.get('verbose', False)
            kwargs_to_pass = {}
            if 'include_edge_tokens' in kwargs:
                kwargs_to_pass['include_edge_tokens'] = kwargs['include_edge_tokens']
            if 'omit_most_frequent_edge' in kwargs:
                kwargs_to_pass['omit_most_frequent_edge'] = kwargs['omit_most_frequent_edge']
            return FCPPSerializer(verbose=verbose, **kwargs_to_pass)
        elif serializer_type == "image_row_major":
            return ImageRowMajorSerializer()
        elif serializer_type == "image_serpentine":
            return ImageSerpentineSerializer()
        elif serializer_type == "image_diag_zigzag":
            return ImageDiagZigzagSerializer()
        else:
            raise ValueError(f"未知序列化器类型: {serializer_type}")
    
    @staticmethod
    def get_available_serializers() -> list:
        """获取可用的序列化器类型"""
        return [
            "smiles", 
            "dfs", "bfs", "eulerian", "topo", "feuler", "cpp", "fcpp",
        ]
    @staticmethod
    def get_image_serializers() -> list:
        """获取可用的图像序列化器类型"""
        return [
            "image_row_major", "image_serpentine", "image_diag_zigzag"
        ]