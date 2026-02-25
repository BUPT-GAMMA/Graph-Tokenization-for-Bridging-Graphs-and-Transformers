"""QM9Test data loader. A small-scale QM9 subset (10%) for testing."""

import time
import numpy as np
from typing import Dict, Optional, Any
import json

# Required dependencies

from .qm9_loader import QM9Loader
from config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class QM9TestLoader(QM9Loader):
    """QM9Test loader - uses 10% of QM9 data."""
    
    def __init__(self, config: ProjectConfig, target_property: Optional[str] = None):
        # Override dataset name, then call parent init
        self.dataset_name = "qm9test"
        super().__init__(config, self.dataset_name, target_property)
        
        # Use config data root; BaseDataLoader sets data_dir = config.data_dir/dataset_name
        if not self.data_dir.exists():
            raise FileNotFoundError(f"QM9Test dataset directory not found: {self.data_dir}")
        logger.info(f"Initializing QM9Test loader: {self.data_dir}")
    
    def _get_data_metadata(self) -> Dict[str, Any]:
        # Ensure data is loaded
        if self._train_data is None:
            self.load_data()
        
        all_data = self._train_data + self._val_data + self._test_data
        
        if not all_data:
            return {}
        
        # Statistics
        num_samples = len(all_data)
        
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
        
        # Load QM9Test-specific metadata
        metadata_file = self.data_dir / "metadata.json"
        qm9test_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    qm9test_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load QM9Test metadata: {e}")
        
        metadata = {
            'dataset_name': 'qm9test',
            'dataset_type': 'molecular_graph',
            'data_source': 'qm9_subset',
            'total_molecules': num_samples,
            'target_property': self.target_property,
            'qm9_properties': self.QM9_PROPERTIES,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_dir': str(self.data_dir),
            'property_availability': property_stats,
            'split_ratios': {
                'train': self.TRAIN_RATIO,
                'val': self.VAL_RATIO,
                'test': self.TEST_RATIO
            },
            # QM9Test-specific info
            'source_dataset': qm9test_metadata.get('source_dataset', 'qm9'),
            'test_ratio': qm9test_metadata.get('test_ratio', 0.1),
            'original_indices': qm9test_metadata.get('original_indices', []),
            'creation_time': qm9test_metadata.get('creation_time', 'Unknown'),
            'random_state': qm9test_metadata.get('random_state', 42)
        }
        
        return metadata

