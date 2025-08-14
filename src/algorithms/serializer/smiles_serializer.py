"""
SMILES序列化器
"""

from typing import Dict, Any, List, Tuple
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping


class SMILESSerializer(BaseGraphSerializer):
    """SMILES字符串序列化器"""
    
    def __init__(self, smiles_type: str = "smiles_1"):
        super().__init__()
        self.name = "smiles"
        self.smiles_type = smiles_type
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """
        初始化SMILES序列化器（统一接口实现）
        
        Args:
            dataset_loader: 数据集加载器
            graph_data_list: 图数据列表（SMILES不需要统计信息）
        """
        # 保存数据集加载器引用
        self._dataset_loader = dataset_loader
        assert self._dataset_loader.dataset_name in ['qm9test','qm9','zinc','aqsol'], f"SMILES序列化器不支持{self._dataset_loader.dataset_name}数据集"
                
        
        # SMILES序列化器不需要预处理统计信息
        self._dataset_stats.update({
            'method': 'SMILES',
            'description': 'SMILES字符串序列化',
            'requires_statistics': False
        })
        
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """
        序列化单个图（统一接口实现）
        
        Args:
            graph_data: 图数据，包含dgl_graph等字段
            **kwargs: 额外的序列化参数
            
        Returns:
            SerializationResult: 序列化结果
        """
        # 调用新的序列化逻辑，返回token和element序列
        token_sequence, element_sequence = self._serialize_with_elements(graph_data)
        
        # 构建SerializationResult
        dgl_graph = graph_data.get('dgl_graph')
        if dgl_graph is None:
            # 如果没有DGL图，创建一个空的
            import dgl
            dgl_graph = dgl.graph([])
        
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _serialize_with_elements(self, graph_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """
        使用SMILES序列化图，返回token序列和element序列
        
        Args:
            graph_data: 图数据，包含smiles或dgl_graph字段
            
        Returns:
            Tuple[List[int], List[str]]: (token序列, element序列)
        """
        # 尝试获取SMILES字符串
        smiles = graph_data[self.smiles_type]
        token_sequence = [ord(c) for c in smiles]
        element_sequence = [f"char_{i}_{c}" for i, c in enumerate(smiles)]
        
        return token_sequence, element_sequence
    
    # def _graph_to_smiles(self, dgl_graph) -> str:
    #     """
    #     从DGL图生成SMILES字符串（简化实现）
        
    #     Args:
    #         dgl_graph: DGL图
            
    #     Returns:
    #         str: SMILES字符串
    #     """
    #     # 这里是一个简化的实现，实际应用中可能需要更复杂的分子图到SMILES的转换
    #     # 对于分子图，我们可以尝试从节点特征重建SMILES
        
    #     if dgl_graph.num_nodes() == 0:
    #         return ""
        
    #     # 检查是否有原子类型信息
    #     if 'atomic_num' in dgl_graph.ndata:
    #         atomic_nums = dgl_graph.ndata['atomic_num']
    #         atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
            
    #         # 简单的线性SMILES生成（仅用于演示）
    #         smiles_parts = []
    #         for atomic_num in atomic_nums:
    #             symbol = atom_symbols.get(int(atomic_num.item()), f'[{int(atomic_num.item())}]')
    #             smiles_parts.append(symbol)
            
    #         return ''.join(smiles_parts)
    #     else:
    #         # 如果没有原子信息，返回一个占位符
    #         return f"UNKNOWN_{dgl_graph.num_nodes()}"
    

    
 