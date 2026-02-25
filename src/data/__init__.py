"""Data layer - unified graph dataset interface."""

# from .qm9_loader import QM9Loader  # avoid forcing DGL import at package level

# Lazy-import heavy DGL deps only when needed
# from .single_graph_loader import SingleGraphLoader, load_single_graph_dataset

# Unified data loader factory (recommended)
from .unified_data_factory import (
    UnifiedDataFactory,
    get_dataloader,
    get_dataset,
    list_available_datasets,
    get_dataset_info
)

# Unified data interface
# Note: do not import UnifiedDataInterface here to avoid circular imports
# Use: from src.data.unified_data_interface import UnifiedDataInterface

# Legacy preprocessed data interface (deprecated; use UnifiedDataInterface)
# Kept temporarily for backward compatibility

# Public API
__all__ = [
    # Unified data loader factory (recommended)
    'UnifiedDataFactory',
    'get_dataloader',
    'get_dataset',
    'list_available_datasets',
    'get_dataset_info',
    
    # Unified data interface (recommended)
    # 'UnifiedDataInterface',  # not exported here to avoid circular imports
]
