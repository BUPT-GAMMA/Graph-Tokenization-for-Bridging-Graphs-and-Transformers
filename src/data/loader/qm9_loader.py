"""QM9 data loader. Reads preprocessed QM9 dataset with multi-task labels and multiple SMILES formats.
QM9数据加载器。读取预处理的QM9数据集，支持多任务标签和多种SMILES格式。"""

 
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


class QM9Loader(BaseDataLoader):
    """QM9 data loader with multi-task support and dataset splits."""
    
    # QM9 dataset info
    QM9_PROPERTIES = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", 
        "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", 
        "h298_atom", "g298_atom"
    ]
    BOND_TYPES = {0:'NONE', 1:'SINGLE', 2:'DOUBLE', 3:'TRIPLE', 4:'AROMATIC'}
    ATOM_TYPES = {1:'H', 2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",57:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",71:"Lu",72:"Hf",73:"Ta",74:"W",75:"Re",76:"Os",77:"Ir",78:"Pt",79:"Au",80:"Hg",81:"Tl",82:"Pb",83:"Bi",84:"Po",85:"At",86:"Rn",87:"Fr",88:"Ra",89:"Ac",90:"Th",91:"Pa",92:"U",93:"Np",94:"Pu",95:"Am",96:"Cm",97:"Bk",98:"Cf",99:"Es",100:"Fm",101:"Md",102:"No",103:"Lr",104:"Rf",105:"Db",106:"Sg",107:"Bh",108:"Hs",109:"Mt",110:"Ds",111:"Rg",112:"Cn",113:"Nh",114:"Fl",115:"Mc",116:"Lv",117:"Ts",118:"Og"} 
    
    
    
    def __init__(self, config: ProjectConfig, dataset_name: str='qm9',
                 target_property: Optional[str] = None):
        super().__init__(dataset_name, config, target_property)
        
        # Attribute cache (in-memory only)
        self._node_attr_cache = {}  # id(graph) -> {node_id -> attr_value}
        self._edge_attr_cache = {}  # id(graph) -> {edge_id -> attr_value}
        self._cache_built = False
        
        # All data cache
        self._all_data = None
        self.load_data()
    
    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"Loading QM9 data from: {self.data_dir}")
        
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
                sample = {
                    'id': f"molecule_{i}",
                    'dgl_graph': graph,
                    'num_nodes': graph.num_nodes(),
                    'num_edges': graph.num_edges() // 2,  # DGL stores bidirectional edges
                    'properties': label,
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
            properties = sample['properties']
            
            if self.target_property:
                # Specified target -> single label
                if self.target_property in properties:
                    label = properties[self.target_property]
                    labels.append(label)
                else:
                    logger.warning(f"Target property {self.target_property} not found in sample")
                    labels.append(0.0)
            else:
                # No target specified -> return all properties (multi-label)
                labels.append(properties)
        
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
            for prop_name in self.QM9_PROPERTIES:
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
            'data_source': 'preprocessed_qm9',
            'total_molecules': num_samples,
            'avg_num_nodes': int(np.mean(num_nodes_list)),
            'avg_num_edges': int(np.mean(num_edges_list)),
            'target_property': self.target_property,
            'qm9_properties': self.QM9_PROPERTIES,
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
        """QM9 is a regression task (multi-property)."""
        return "regression"

    def get_default_target_property(self) -> str:
        """Return a commonly used default property."""
        return self.QM9_PROPERTIES[0]
    
    def get_downstream_label_keys(self) -> List[str]:
        """Return available label property keys."""
        return self.QM9_PROPERTIES.copy()
    
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

            # ========= Extract type IDs as tensors =========
            # Node atomic number: ndata['attr'] column 6 (index 5)
            assert 'attr' in dgl_graph.ndata, "Graph missing ndata['attr']"
            attr_features = dgl_graph.ndata['attr']  # [N, 12]
            atomic_num_tensor = attr_features[:, 5].long().clamp(min=0)
            dgl_graph.ndata['node_type_id'] = atomic_num_tensor

            # Edge bond type: argmax of first 4 dims of edata['edge_attr'], +1 to unified space
            assert 'edge_attr' in dgl_graph.edata, "Graph missing edata['edge_attr']"
            edge_attrs = dgl_graph.edata['edge_attr']  # [E, 4]
            bond_type_base = torch.argmax(edge_attrs[:, :4], dim=1).long()  # 0..3
            edge_type_ids = bond_type_base + 1  # 1..4 (0 reserved for NONE)
            dgl_graph.edata['edge_type_id'] = edge_type_ids

            # ========= Build in-memory cache for single-item API =========
            self._node_attr_cache[graph_id] = {int(i): int(v) for i, v in enumerate(atomic_num_tensor.tolist())}
            self._edge_attr_cache[graph_id] = {int(i): int(v) for i, v in enumerate(edge_type_ids.tolist())}

            # ========= Write fixed-length token tensors =========
            # QM9 token dim=1: node token = 2*atomic_num+1; edge token = 2*edge_type_id
            node_token_ids = (atomic_num_tensor * 2 + 1).view(-1, 1)
            edge_token_ids = (edge_type_ids * 2).view(-1, 1)
            dgl_graph.ndata['node_token_ids'] = node_token_ids
            dgl_graph.edata['edge_token_ids'] = edge_token_ids
        
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
                node_cache = self._node_attr_cache[graph_id]
                if node_id in node_cache:
                    return node_cache[node_id]
        
        # Fallback
        if 'attr' in graph.ndata:
            attr_features = graph.ndata['attr']
            if node_id < len(attr_features):
                atomic_num = int(attr_features[node_id][5].item())
                return atomic_num
        
        raise ValueError(f"Cannot get attribute for node {node_id}")
    
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> int:
        assert 0 <= edge_id < graph.num_edges(), f"Edge ID {edge_id} out of range"
        
        # Use cache first
        if self._cache_built:
            graph_id = id(graph)
            if graph_id in self._edge_attr_cache:
                edge_cache = self._edge_attr_cache[graph_id]
                if edge_id in edge_cache:
                    return edge_cache[edge_id]
        
        # Fallback: extract from one-hot encoding
        if 'edge_attr' in graph.edata:
            edge_attrs = graph.edata['edge_attr']
            if edge_id < len(edge_attrs):
                bond_type = torch.argmax(edge_attrs[edge_id][:4]).item()
                return bond_type
        
        raise ValueError(f"Cannot get attribute for edge {edge_id}")
    
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        return self.ATOM_TYPES[self.get_node_attribute(graph, node_id)]
    
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        return self.BOND_TYPES[self.get_edge_attribute(graph, edge_id)]

    # ==================== Bulk tensor interface ====================

    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[str]:
        assert 'node_type_id' in graph.ndata, "Missing node_type_id"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        type_ids = graph.ndata['node_type_id'][ids].tolist()
        return [self.ATOM_TYPES[int(tid)] for tid in type_ids]

    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[str]:
        assert 'edge_type_id' in graph.edata, "Missing edge_type_id"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        type_ids = graph.edata['edge_type_id'][ids].tolist()
        return [self.BOND_TYPES[int(tid)] for tid in type_ids]

    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: List[int]) -> List[List[int]]:
        assert 'node_token_ids' in graph.ndata, "Missing node_token_ids"
        ids = torch.as_tensor(node_ids, dtype=torch.long)
        tok = graph.ndata['node_token_ids'][ids]  # [N, D]
        return tok.tolist()

    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: List[int]) -> List[List[int]]:
        assert 'edge_token_ids' in graph.edata, "Missing edge_token_ids"
        ids = torch.as_tensor(edge_ids, dtype=torch.long)
        tok = graph.edata['edge_token_ids'][ids]
        return tok.tolist()
      
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        node_key = ('atom', self.get_node_attribute(graph, node_id, ntype))
        return [self.token_map[node_key]]
      
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        edge_key = ('bond', self.get_edge_attribute(graph, edge_id, etype))
        return [self.token_map[edge_key]]
    
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

    # ==================== Whole-graph tensor interface ====================
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'node_type_id' in graph.ndata, "Missing node_type_id"
        return graph.ndata['node_type_id']

    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'edge_type_id' in graph.edata, "Missing edge_type_id"
        return graph.edata['edge_type_id']

    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'node_token_ids' in graph.ndata, "Missing node_token_ids"
        return graph.ndata['node_token_ids']

    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        assert 'edge_token_ids' in graph.edata, "Missing edge_token_ids"
        return graph.edata['edge_token_ids']

    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        return graph.edges()

    def get_edge_type_id_by_name(self, name: str) -> int:
        inv = {v: k for k, v in self.BOND_TYPES.items()}
        assert name in inv, f"Unknown edge type: {name}"
        return int(inv[name])
    