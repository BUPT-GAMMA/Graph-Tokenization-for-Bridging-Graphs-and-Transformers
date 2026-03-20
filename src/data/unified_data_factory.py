"""Unified data loader factory. Manages all dataset loaders with a single creation/access interface.
统一数据加载器工厂。通过单一创建/访问接口管理所有数据集加载器。"""

from typing import Dict, Type, Optional, Any, List, Tuple

from config import ProjectConfig
# Lazy imports to avoid loading all deps (especially DGL/OGB) at module import
from .base_loader import BaseDataLoader
# from .qm9_loader import QM9Loader
# from .qm9test_loader import QM9TestLoader  
# from .zinc_loader import ZINCLoader
# from .aqsol_loader import AQSOLoader
# from .mnist_loader import MNISTDataLoader
# from .loaders.cora_loader_v2 import CoraLoaderV2
from utils.logger import get_logger

logger = get_logger(__name__)


class UnifiedDataFactory:
    """Unified data loader factory."""
    
    _registry: Dict[str, Type[BaseDataLoader]] = {}
    _instances: Dict[str, BaseDataLoader] = {}
    
    @classmethod
    def register(cls, name: str, loader_class_or_factory) -> None:
        """Register a data loader class or factory function."""
        # Supports direct class or lazy-import factory function
        cls._registry[name] = loader_class_or_factory
        # logger.info(f"Registered loader: {name} -> {getattr(loader_class_or_factory, '__name__', 'factory_function')}")
    
    @classmethod
    def create(cls, dataset_name: str, config: ProjectConfig, 
               target_property: Optional[str] = None,
               **kwargs) -> BaseDataLoader:
        """Create a data loader instance."""
        if dataset_name not in cls._registry:
            available_datasets = list(cls._registry.keys())
            raise ValueError(f"Unknown dataset: {dataset_name}. Registered: {available_datasets}")
        
        # Check for cached instance (considering target_property)
        cache_key = f"{dataset_name}_{target_property}_{id(config)}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # Create new instance - supports lazy import
        loader_class_or_factory = cls._registry[dataset_name]
        
        # If factory function, call it to get actual class
        if callable(loader_class_or_factory) and not hasattr(loader_class_or_factory, '__mro__'):
            loader_class = loader_class_or_factory()
        else:
            loader_class = loader_class_or_factory
            
        # Validate subclass
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader {loader_class} must inherit from BaseDataLoader")
            
        loader = loader_class(config, target_property=target_property, **kwargs)
        
        # Cache instance
        cls._instances[cache_key] = loader
        
        logger.info(f"Created loader: {dataset_name} (target: {target_property})")
        return loader
    
    @classmethod
    def list_datasets(cls) -> Dict[str, Type[BaseDataLoader]]:
        """Return all registered datasets."""
        return cls._registry.copy()
    
    @classmethod
    def get_loader_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Return loader info for a dataset."""
        if dataset_name not in cls._registry:
            return {}
        
        loader_class = cls._registry[dataset_name]
        return {
            'name': dataset_name,
            'class': loader_class.__name__,
            'module': loader_class.__module__,
            'doc': loader_class.__doc__
        }
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached loader instances."""
        cls._instances.clear()
        logger.info("Cleared all loader cache")


def _lazy_import_loader(loader_name: str):
    """Lazy-import a loader to avoid loading all deps at factory init."""
    if loader_name == "qm9":
        from .loader.qm9_loader import QM9Loader
        return QM9Loader
    elif loader_name == "qm9test":
        from .loader.qm9test_loader import QM9TestLoader
        return QM9TestLoader
    elif loader_name == "zinc":
        from .loader.zinc_loader import ZINCLoader
        return ZINCLoader
    elif loader_name == "aqsol":
        from .loader.aqsol_loader import AQSOLoader
        return AQSOLoader
    elif loader_name == "mnist":
        from .loader.mnist_loader import MNISTDataLoader
        return MNISTDataLoader
    elif loader_name == "mnist_raw":
        from .loader.mnist_raw_loader import MNISTRawDataLoader
        return MNISTRawDataLoader
    elif loader_name == "colors3":
        from .loader.colors3_loader import COLORS3Loader
        return COLORS3Loader
    elif loader_name == "proteins":
        from .loader.proteins_loader import PROTEINSLoader
        return PROTEINSLoader
    elif loader_name == "synthetic":
        from .loader.synthetic_loader import SYNTHETICLoader
        return SYNTHETICLoader
    elif loader_name == "mutagenicity":
        from .loader.mutagenicity_loader import MutagenicityLoader
        return MutagenicityLoader
    elif loader_name == "coildel":
        from .loader.coildel_loader import COILDELLoader
        return COILDELLoader
    elif loader_name == "dblp":
        from .loader.dblp_loader import DBLPLoader
        return DBLPLoader
    elif loader_name == "dd":
        from .loader.dd_loader import DDLoader
        return DDLoader
    elif loader_name == "twitter":
        from .loader.twitter_loader import TwitterLoader
        return TwitterLoader
    elif loader_name == "molhiv":
        from .loader.molhiv_loader import MOLHIVLoader
        return MOLHIVLoader
    elif loader_name == "peptides_func":
        from .loader.peptides_func_loader import PeptidesFuncLoader
        return PeptidesFuncLoader
    elif loader_name == "peptides_struct":
        from .loader.peptides_struct_loader import PeptidesStructLoader
        return PeptidesStructLoader
    elif loader_name == "code2":
        from .loader.code2_loader import CODE2Loader
        return CODE2Loader
    else:
        raise ValueError(f"Unknown loader: {loader_name}")

def _register_all_loaders():
    """Register all loaders using lazy import."""
    loader_names = ["qm9", "qm9test", "zinc", "aqsol", "mnist", "mnist_raw", "colors3", "proteins", "synthetic", "mutagenicity", "coildel", "dblp", "dd", "twitter", "molhiv", "peptides_func", "peptides_struct", "code2"]
    
    for name in loader_names:
        # Register lazy-import factory instead of direct class import
        UnifiedDataFactory.register(name, lambda n=name: _lazy_import_loader(n))
    
    logger.info("All loaders registered")


# Auto-register all loaders
_register_all_loaders()


def get_dataloader(dataset_name: str, config: ProjectConfig, 
                   target_property: Optional[str] = None,
                   **kwargs) -> BaseDataLoader:
    """Get a data loader instance."""
    return UnifiedDataFactory.create(dataset_name, config, target_property, **kwargs)


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(UnifiedDataFactory.list_datasets().keys())


def get_dataset(dataset_name: str, config: ProjectConfig, 
                target_property: Optional[str] = None,
                **kwargs) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                                  List[Any], List[Any], List[Any]]:
    """Get dataset splits and labels."""
    loader = get_dataloader(dataset_name, config, target_property, **kwargs)
    return loader.load_data()


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get dataset info."""
    return UnifiedDataFactory.get_loader_info(dataset_name)

