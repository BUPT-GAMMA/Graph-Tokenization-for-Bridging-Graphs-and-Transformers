"""ZINC data loader. Reads preprocessed ZINC dataset with multi-task labels and multiple SMILES formats.
ZINC数据加载器。读取预处理的ZINC数据集，支持多任务标签和多种SMILES格式。"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

# Required dependencies
import dgl
import torch

from ..base_loader import BaseDataLoader
from config import ProjectConfig
from utils.logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


class ZINCLoader(BaseDataLoader):
    """ZINC data loader with multi-task support and dataset splits."""
    
    # ZINC dataset info
    ZINC_PROPERTIES = ["logP_SA_cycle_normalized"]
    
    ATOM_TYPES = {1:'H', 2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",91:"Pa",92:"U",93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",104:"Rf",105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og"}
    BOND_TYPES = {0:'NONE', 1:'SINGLE', 2:'DOUBLE', 3:'TRIPLE', 4:'AROMATIC'}
    
    def __init__(self, config: ProjectConfig, 
                 target_property: Optional[str] = None):
        super().__init__("zinc", config, target_property)
        
        # Attribute cache (in-memory only)
        self._node_attr_cache = {}  # id(graph) -> {node_id -> attr_value}
        self._edge_attr_cache = {}  # id(graph) -> {edge_id -> attr_value}
        self._cache_built = False
        
        # All data cache
        self._all_data = None
        self.load_data()
    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"Loading ZINC data from: {self.data_dir}")
        
        # Check index files
        train_index_file = self.data_dir / "train_index.json"
        test_index_file = self.data_dir / "test_index.json"
        val_index_file = self.data_dir / "val_index.json"
        
        if not all(f.exists() for f in [train_index_file, test_index_file, val_index_file]):
            raise FileNotFoundError("Index files not found; run preprocessing first")
        
        # Load all data
        data_file = self.data_dir / "data.pkl"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load unified data file
        logger.info("Loading unified data file...")
        if self._all_data is None:
            all_data = self._load_data_file(data_file)
            self._all_data = all_data
        all_data = self._all_data
        
        # Load 4 SMILES formats
        logger.info("Loading SMILES data...")
        all_data = self._add_smiles_to_all_data(all_data)
        
        logger.info(f"All data loaded: {len(all_data)} samples")
        
        # Load split indices
        logger.info("Loading split indices...")
        split_indices = self.get_split_indices()
        
        # Split data by indices
        train_indices = split_indices['train']
        test_indices = split_indices['test']
        val_indices = split_indices['val']
        
        # Extract data by indices
        train_data = [all_data[i] for i in train_indices if i < len(all_data)]
        test_data = [all_data[i] for i in test_indices if i < len(all_data)]
        val_data = [all_data[i] for i in val_indices if i < len(all_data)]
        
        logger.info(f"Data loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def _add_smiles_to_all_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add 4 SMILES formats to all data samples."""
        # SMILES format filenames
        smiles_files = {
            "smiles_1": "smiles_1_direct.txt",
            "smiles_2": "smiles_2_explicit_h.txt", 
            "smiles_3": "smiles_3_addhs.txt",
            "smiles_4": "smiles_4_addhs_explicit_h.txt"
        }
        
        # Load each SMILES format
        smiles_data = {}
        for key, filename in smiles_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        smiles_list = [line.strip() for line in f.readlines()]
                    smiles_data[key] = smiles_list
                    logger.debug(f"Loaded {filename}: {len(smiles_list)} SMILES")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")
                    smiles_data[key] = []
            else:
                logger.warning(f"SMILES file not found: {file_path}")
                smiles_data[key] = []
        
        # Add SMILES to each sample (in order)
        for i, sample in enumerate(data):
            for key, smiles_list in smiles_data.items():
                if i < len(smiles_list):
                    sample[key] = smiles_list[i]
                else:
                    sample[key] = ""
        
        return data
    
    def _load_data_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single data file."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)

            # Convert tuple format to dict format
            data = []
            for i, (graph, label) in enumerate(raw_data):
                # Convert torch.Tensor labels to dict format
                if isinstance(label, torch.Tensor):
                    # Single scalar -> dict
                    if label.numel() == 1:
                        properties = {self.ZINC_PROPERTIES[0]: float(label.item())}
                    else:
                        raise ValueError(f"Bad label format: {label}")
                else:
                    # Already dict format
                    properties = label
                
                sample = {
                    'id': f"molecule_{i}",
                    'dgl_graph': graph,
                    'num_nodes': graph.num_nodes(),
                    'num_edges': graph.num_edges() // 2,  # DGL stores bidirectional edges
                    'properties': properties,
                    'dataset_name': self.dataset_name,
                    'data_type': 'molecular_graph'
                }
                data.append(sample)
            
            return data
        except Exception as e:
            logger.error(f"Failed to load data file {file_path}: {e}")
            raise
    
    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[Any]:
        labels = []
        
        for sample in data:
            label = sample['properties']['logP_SA_cycle_normalized']
            labels.append(label)
        
        return labels
    
    def _get_data_metadata(self) -> Dict[str, Any]:
        # Ensure data is loaded
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        # Statistics
        num_samples = len(all_data)
        num_nodes_list = [sample.get('num_nodes', 0) for sample in all_data]
        num_edges_list = [sample.get('num_edges', 0) for sample in all_data]
        
        # Property statistics
        property_stats = {}
        if all_data and 'properties' in all_data[0]:
            properties = all_data[0]['properties']
            for prop_name in self.ZINC_PROPERTIES:
                if prop_name in properties:
                    prop_values = [sample.get('properties', {}).get(prop_name) for sample in all_data 
                                 if prop_name in sample.get('properties', {})]
                    if prop_values:
                        property_stats[prop_name] = {
                            'min': float(np.min(prop_values)),
                            'max': float(np.max(prop_values)),
                            'mean': float(np.mean(prop_values)),
                            'std': float(np.std(prop_values))
                        }
        
        metadata = {
            'dataset_name': self.dataset_name,
            'dataset_type': 'molecular_graph',
            'data_source': 'preprocessed_zinc',
            'total_molecules': num_samples,
            'avg_num_nodes': int(np.mean(num_nodes_list)),
            'avg_num_edges': int(np.mean(num_edges_list)),
            'target_property': self.target_property,
            'zinc_properties': self.ZINC_PROPERTIES,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': str(self.data_dir),
            'property_availability': property_stats,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO
            }
        }
        
        return metadata

    # ---------------- Downstream task info ----------------
    def get_dataset_task_type(self) -> str:
        """ZINC is typically a single-property regression task."""
        return "regression"

    def get_default_target_property(self) -> str:
        """Return default regression property key."""
        return self.ZINC_PROPERTIES[0]
    
    def get_downstream_label_keys(self) -> List[str]:
        """Return available label property keys."""
        return self.ZINC_PROPERTIES.copy()
    
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                           List[Any], List[Any], List[Any]]:
        """Override load_data to build attribute cache."""
        res = super().load_data()
        # Build attribute cache
        if self._all_data is not None:
            self._build_attribute_cache(self._all_data)
        return res
    
    def _build_attribute_cache(self, processed_data: List[Dict[str, Any]]) -> None:
        """Build node and edge attribute caches."""
        import time
        start_time = time.time()
        logger.info("Building attribute cache...")
        
        for sample in tqdm(processed_data, desc="Building cache"):
            dgl_graph = sample['dgl_graph']
            graph_id = id(dgl_graph)
            
            # Build node attribute cache
            # ZINC format: one discrete value per node (atomic number)
            node_cache = {}
            if 'feat' in dgl_graph.ndata:
                node_features = dgl_graph.ndata['feat']
                for node_id in range(dgl_graph.num_nodes()):
                    if node_id < len(node_features):
                        atomic_num = int(node_features[node_id].item())
                        node_cache[node_id] = atomic_num
            self._node_attr_cache[graph_id] = node_cache
            
            # Build edge attribute cache
            # ZINC format: one discrete value per edge (bond type)
            edge_cache = {}
            if 'feat' in dgl_graph.edata:
                edge_features = dgl_graph.edata['feat']
                for edge_id in range(dgl_graph.num_edges()):
                    if edge_id < len(edge_features):
                        bond_type = int(edge_features[edge_id].item())
                        edge_cache[edge_id] = bond_type
            self._edge_attr_cache[graph_id] = edge_cache
        
        self._cache_built = True
        cache_time = time.time() - start_time
        
        # Cache statistics
        total_nodes = sum(len(cache) for cache in self._node_attr_cache.values())
        total_edges = sum(len(cache) for cache in self._edge_attr_cache.values())
        
        logger.info(f"Attribute cache built: {len(self._node_attr_cache)} graphs (nodes), {len(self._edge_attr_cache)} graphs (edges)")
        logger.info(f"Cache stats: {total_nodes} node attrs, {total_edges} edge attrs")
        logger.info(f"Cache build time: {cache_time:.2f}s")
    
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> int:
        assert 0 <= node_id < graph.num_nodes(), f"Node ID {node_id} out of range"
        
        # Use cache first
        if self._cache_built:
            graph_id = id(graph)
            if graph_id in self._node_attr_cache:
                cache = self._node_attr_cache[graph_id]
                if node_id in cache:
                    return cache[node_id]
        
        # Fallback
        if 'feat' in graph.ndata:
            node_features = graph.ndata['feat']
            if node_id < len(node_features):
                return int(node_features[node_id].item())
        
        raise ValueError(f"Cannot get attribute for node {node_id}")
    
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> int:
        assert 0 <= edge_id < graph.num_edges(), f"Edge ID {edge_id} out of range"
        
        # Use cache first
        if self._cache_built:
            graph_id = id(graph)
            if graph_id in self._edge_attr_cache:
                cache = self._edge_attr_cache[graph_id]
                if edge_id in cache:
                    return cache[edge_id]
        
        # Fallback
        if 'feat' in graph.edata:
            edge_features = graph.edata['feat']
            if edge_id < len(edge_features):
                return int(edge_features[edge_id].item())
        
        raise ValueError(f"Cannot get attribute for edge {edge_id}")
    
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        return self.ATOM_TYPES[self.get_node_attribute(graph, node_id)]
    
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return self.BOND_TYPES[self.get_edge_attribute(graph, edge_id)]
      
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        return [self.token_map[('atom', self.get_node_attribute(graph, node_id, ntype))]]
      
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        return [self.token_map[('bond', self.get_edge_attribute(graph, edge_id, etype))]]
      
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        
        node_token_map = {}
        # Atomic numbers -> odd tokens
        for atomic_num in range(1, 119):
            odd_token = 2 * atomic_num + 1
            node_token_map[('atom', atomic_num)] = odd_token
            
        edge_token_map = {}
        # Bond types -> even tokens
        for bond_type_id, bond_type_name in self.BOND_TYPES.items():
            even_token = 2 * bond_type_id
            edge_token_map[('bond', bond_type_id)] = even_token
            
        return {**node_token_map, **edge_token_map}
    
    def get_token_readable(self, token_id: int) -> str:
        
        if token_id % 2 == 0:
            return {0:'',1:"-",2:"=",3:"*",4:"@"}[token_id // 2]
        else:
            return self.ATOM_TYPES[token_id // 2]
      
    def get_most_frequent_edge_type(self) -> str:
        return 'SINGLE'

    # ==================== Bulk & whole-graph tensor interface ====================
    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[str]:
        assert 'feat' in graph.ndata, "Missing node feat"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        type_ids = graph.ndata['feat'][ids].tolist()
        return [self.ATOM_TYPES[int(tid)] for tid in type_ids]

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[str]:
        assert 'feat' in graph.edata, "Missing edge feat"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        type_ids = graph.edata['feat'][ids].tolist()
        return [self.BOND_TYPES[int(tid)] for tid in type_ids]

    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[List[int]]:
        assert 'feat' in graph.ndata, "Missing node feat"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        atomic = graph.ndata['feat'][ids].long()
        tok = (atomic * 2 + 1).view(-1, 1)
        return tok.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[List[int]]:
        assert 'feat' in graph.edata, "Missing edge feat"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        bond = graph.edata['feat'][ids].long()
        tok = (bond * 2).view(-1, 1)
        return tok.tolist()

    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'feat' in graph.ndata, "Missing node feat"
        return graph.ndata['feat'].long()

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'feat' in graph.edata, "Missing edge feat"
        return graph.edata['feat'].long()

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        nt = self.get_graph_node_type_ids(graph)
        return (nt * 2 + 1).view(-1, 1)

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        et = self.get_graph_edge_type_ids(graph)
        return (et * 2).view(-1, 1)

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_edge_type_id_by_name(self, name: str) -> int:
        inv = {v: k for k, v in self.BOND_TYPES.items()}
        assert name in inv, f"Unknown edge type: {name}"
        return int(inv[name])
    