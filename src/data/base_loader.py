"""Base class for all dataset loaders. Provides a unified interface and shared logic."""

import os
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence
import dgl
import numpy as np
import torch
# Note: tqdm not imported here; subclasses may import as needed

from config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseDataLoader(ABC):
    """Base class for all dataset loaders."""
    
    # Fixed dataset split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    def __init__(self, dataset_name: str, config: ProjectConfig, 
                 target_property: Optional[str] = None):
        self.dataset_name = dataset_name
        self.config = config
        self.target_property = target_property
        
        # Data path (uses global config data_dir)
        self.data_dir = Path(self.config.data_dir) / self.dataset_name
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        # In-memory cache
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._train_labels = None
        self._val_labels = None
        self._test_labels = None
        self._metadata = None
        self.token_map = None
        self.token_readable = None
        self._all_data = None
    @abstractmethod
    def _load_processed_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load data from preprocessed directory (subclass must implement).

        Returns:
            (train_data, val_data, test_data)
        """
        pass
    
    @abstractmethod
    def _extract_labels(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Extract labels from data (subclass must implement)."""
        pass
    
    @abstractmethod
    def _get_data_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata (subclass must implement)."""
        pass

    # ==================== Token interface (subclass must implement) ====================
    
    @abstractmethod
    def get_node_attribute(self, graph: dgl.DGLGraph, node_id: int) -> int:
        """Return the key attribute of a node (e.g. atomic number) for token mapping."""
        pass
    
    @abstractmethod
    def get_edge_attribute(self, graph: dgl.DGLGraph, edge_id: int) -> int:
        """Return the key attribute of an edge (e.g. bond type) for token mapping."""
        pass
    
    @abstractmethod
    def get_node_type(self, graph: dgl.DGLGraph, node_id: int) -> str:
        """Return the type string of a node."""
        pass
    
    @abstractmethod
    def get_edge_type(self, graph: dgl.DGLGraph, edge_id: int) -> str:
        """Return the type string of an edge."""
        pass
      
    @abstractmethod
    def get_most_frequent_edge_type(self) -> str:
        """Return the most frequent edge type."""
        pass

    @abstractmethod
    def get_edge_type_id_by_name(self, name: str) -> int:
        """Map edge type name to numeric ID."""
        pass
      
    @abstractmethod
    def get_node_token(self, graph: dgl.DGLGraph, node_id: int, ntype: str = None) -> List[int]:
        """Return the token list for a node."""
        pass
    
    @abstractmethod
    def get_edge_token(self, graph: dgl.DGLGraph, edge_id: int, etype: str = None) -> List[int]:
        """Return the token list for an edge."""
        pass
    
    # ==================== Bulk API (subclass must implement) ====================
    
    @abstractmethod
    def get_node_tokens_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[List[int]]:
        """Bulk get node tokens (subclass must implement efficient version)."""
        pass

    @abstractmethod
    def get_edge_tokens_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[List[int]]:
        """Bulk get edge tokens (subclass must implement efficient version)."""
        pass

    @abstractmethod
    def get_node_types_bulk(self, graph: dgl.DGLGraph, node_ids: Sequence[int]) -> List[str]:
        """Bulk get node types (subclass must implement efficient version)."""
        pass

    @abstractmethod
    def get_edge_types_bulk(self, graph: dgl.DGLGraph, edge_ids: Sequence[int]) -> List[str]:
        """Bulk get edge types (subclass must implement efficient version)."""
        pass

    # ==================== Whole-graph tensor interface (subclass must implement) ====================
    @abstractmethod
    def get_graph_node_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Return whole-graph node type IDs, shape [N] (LongTensor)."""
        pass

    @abstractmethod
    def get_graph_edge_type_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Return whole-graph edge type IDs, shape [E] (LongTensor)."""
        pass

    @abstractmethod
    def get_graph_node_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Return whole-graph node tokens, shape [N, Dn] (LongTensor)."""
        pass

    @abstractmethod
    def get_graph_edge_token_ids(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """Return whole-graph edge tokens, shape [E, De] (LongTensor)."""
        pass

    @abstractmethod
    def get_graph_src_dst(self, graph: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return edge (src, dst) index pair (LongTensor)."""
        pass
      
    
    @abstractmethod
    def get_token_map(self) -> Dict[Tuple[str, int], int]:
        """Return dataset-level token mapping."""
        pass
    
    def get_all_data_with_indices(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
        """Return all data and split indices."""
        # Get split indices
        split_indices = self.get_split_indices()
        
        if self._all_data is None:
          self.load_data()
        
        return self._all_data, split_indices
    
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                                 List[Any], List[Any], List[Any]]:
        """Load dataset (unified interface). Returns (train, val, test, train_labels, val_labels, test_labels)."""
        # Return cached data if already loaded
        if self._train_data is not None:
            return (self._train_data, self._val_data, self._test_data, 
                   self._train_labels, self._val_labels, self._test_labels)
        
        logger.info(f"Loading {self.dataset_name} data...")
        logger.info(f"Preprocessed dir: {self.data_dir}")
        
        # Load preprocessed data
        train_data, val_data, test_data = self._load_processed_data()
        logger.info("Preprocessed data loaded; assembling splits...")
        
        if not train_data or not val_data or not test_data:
            raise ValueError(f"Failed to load {self.dataset_name} data")
        
        logger.info(f"Data loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Extract labels
        train_labels = self._extract_labels(train_data)
        val_labels = self._extract_labels(val_data)
        test_labels = self._extract_labels(test_data)
        
        # Cache data
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._train_labels = train_labels
        self._val_labels = val_labels
        self._test_labels = test_labels
        
        logger.info(f"Labels extracted: target_property={self.target_property}")
        self._all_data = train_data + val_data + test_data
        
        logger.info("Building token map...")
        self.token_map = self.get_token_map()
        self.token_readable = {v: k for k, v in self.token_map.items()}
        logger.info("Data loader ready")
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
      
    def get_split_indices(self) -> Dict[str, List[int]]:
        """Return train/val/test split indices."""
        data_dir = self.data_dir
        
        train_index_file = os.path.join(data_dir, "train_index.json")
        val_index_file = os.path.join(data_dir, "val_index.json")
        test_index_file = os.path.join(data_dir, "test_index.json")
        
        try:
            with open(train_index_file, 'r') as f:
                train_indices = json.load(f)
            with open(val_index_file, 'r') as f:
                val_indices = json.load(f)
            with open(test_index_file, 'r') as f:
                test_indices = json.load(f)
            
            return {
                'train': train_indices,
                'test': test_indices,
                'val': val_indices
            }
        except Exception as e:
            raise FileNotFoundError(f"Failed to load index files: {e}")
    
    def get_smiles_data(self) -> Tuple[List[str], List[str], List[str]]:
        """Return SMILES strings for train/val/test splits."""
        raise NotImplementedError(f"Dataset {self.dataset_name} does not support SMILES")
    
    def get_smiles_by_type(self, smiles_type: str = "1") -> Tuple[List[str], List[str], List[str]]:
        """Return SMILES strings of a specific format type."""
        raise NotImplementedError(f"Dataset {self.dataset_name} does not support SMILES")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        if self._metadata is None:
            self._metadata = self._get_data_metadata()
        
        return self._metadata

    # ---------------- Downstream task info (defaults; override in subclass) ----------------
    def get_dataset_task_type(self) -> str:
        """Return default task type. Override in classification datasets."""
        return "regression"

    def get_num_classes(self) -> int:
        """Number of classes for classification. Override in subclass."""
        return 1

    def get_default_target_property(self) -> Optional[str]:
        """Default target property key. Override in subclass."""
        return None

    def get_downstream_label_keys(self) -> List[str]:
        """Return available downstream label keys. Override in subclass."""
        return ['label']

    def get_loss_config(self) -> Optional[Dict[str, Any]]:
        """Return recommended loss config for the dataset. None = use default."""
        return None
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Return data statistics."""
        # Ensure data is loaded
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        stats = {
            'total_samples': len(all_data),
            'train_samples': len(self._train_data),
            'val_samples': len(self._val_data),
            'test_samples': len(self._test_data),
            'dataset_name': self.dataset_name,
            'target_property': self.target_property,
            'cache_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Graph statistics
        num_nodes_list = []
        num_edges_list = []
        
        for sample in all_data:
            if 'dgl_graph' in sample:
                graph = sample['dgl_graph']
                num_nodes_list.append(graph.num_nodes())
                num_edges_list.append(graph.num_edges())
            elif 'num_nodes' in sample:
                num_nodes_list.append(sample['num_nodes'])
            if 'num_edges' in sample:
                num_edges_list.append(sample['num_edges'])
        
        if num_nodes_list:
            stats['graph_statistics'] = {
                'avg_nodes': np.mean(num_nodes_list),
                'std_nodes': np.std(num_nodes_list),
                'min_nodes': np.min(num_nodes_list),
                'max_nodes': np.max(num_nodes_list),
                'total_nodes': sum(num_nodes_list)
            }
        
        if num_edges_list:
            stats['graph_statistics']['avg_edges'] = np.mean(num_edges_list)
            stats['graph_statistics']['std_edges'] = np.std(num_edges_list)
            stats['graph_statistics']['min_edges'] = np.min(num_edges_list)
            stats['graph_statistics']['max_edges'] = np.max(num_edges_list)
            stats['graph_statistics']['total_edges'] = sum(num_edges_list)
        
        return stats
    
    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate whether a data sample is valid."""
        try:
            # Basic field check
            required_fields = ['id']
            for field in required_fields:
                if field not in sample:
                    return False
            
            # Graph data check
            if 'dgl_graph' in sample:
                graph = sample['dgl_graph']
                if not isinstance(graph, dgl.DGLGraph):
                    return False
                if graph.num_nodes() == 0:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_token_readable(self, token_id: int) -> str:
        """Map token ID to human-readable string (optional)."""
        raise NotImplementedError(f"Dataset {self.dataset_name} does not support token-to-readable mapping")

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Compute class weights for imbalanced classification. Returns None for non-classification tasks."""
        if self._train_data is None:
            self.load_data()

        # Check if classification
        task_type = self.get_dataset_task_type()
        if task_type not in ['classification', 'binary_classification']:
            return None

        # Check for training labels
        if not self._train_data:
            return None

        # Count class distribution in training set
        train_labels = self._extract_labels(self._train_data)
        if not train_labels:
            return None

        # Compute class weights
        num_classes = self.get_num_classes()
        class_counts = torch.zeros(num_classes, dtype=torch.float)

        for label in train_labels:
            if isinstance(label, (int, torch.Tensor)):
                label_idx = int(label) if isinstance(label, torch.Tensor) else label
                if 0 <= label_idx < num_classes:
                    class_counts[label_idx] += 1

        # Weight = total_samples / (num_classes * class_count)
        total_samples = len(train_labels)
        class_weights = torch.zeros(num_classes, dtype=torch.float)

        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (num_classes * class_counts[i])
            else:
                # Handle missing classes
                class_weights[i] = 1.0

        return class_weights
      
    # Note: get_most_frequent_edge_type must be implemented by subclass (see abstractmethod above)
        
    def expand_tokens(self, token_lists: List[List[int]]) -> List[int]:
        """Flatten multiple token lists into a single sequence."""
        result = []
        for token_list in token_lists:
            result.extend(token_list)
        return result
