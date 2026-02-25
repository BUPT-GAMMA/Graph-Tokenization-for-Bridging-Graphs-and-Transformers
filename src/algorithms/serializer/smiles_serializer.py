"""SMILES serializer.
SMILES序列化器。"""

from typing import Dict, Any, List, Tuple
from .base_serializer import BaseGraphSerializer, SerializationResult, GlobalIDMapping


class SMILESSerializer(BaseGraphSerializer):
    """SMILES string serializer."""
    
    def __init__(self, smiles_type: str = "smiles_1"):
        super().__init__()
        self.name = "smiles"
        self.smiles_type = smiles_type
        
    def _initialize_serializer(self, dataset_loader, graph_data_list: List[Dict[str, Any]] = None) -> None:
        """Initialize SMILES serializer (no statistics needed)."""
        self._dataset_loader = dataset_loader
        assert self._dataset_loader.dataset_name in ['qm9test','qm9','zinc','aqsol'], f"SMILES serializer does not support dataset {self._dataset_loader.dataset_name}"
                
        
        self._dataset_stats.update({
            'method': 'SMILES',
            'description': 'SMILES string serialization',
            'requires_statistics': False
        })
        
    
    def _serialize_single_graph(self, graph_data: Dict[str, Any], **kwargs) -> SerializationResult:
        """Serialize a single graph."""
        token_sequence, element_sequence = self._serialize_with_elements(graph_data)
        dgl_graph = graph_data.get('dgl_graph')
        if dgl_graph is None:
            import dgl
            dgl_graph = dgl.graph([])
        
        id_mapping = GlobalIDMapping(dgl_graph)
        
        return SerializationResult([token_sequence], [element_sequence], id_mapping)
    
    def _serialize_with_elements(self, graph_data: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """SMILES serialization."""
        smiles = graph_data[self.smiles_type]
        token_sequence = [ord(c) for c in smiles]
        element_sequence = [f"char_{i}_{c}" for i, c in enumerate(smiles)]
        
        return token_sequence, element_sequence
    
    

    
 