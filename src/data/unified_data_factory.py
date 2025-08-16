"""
统一数据加载器工厂
================

管理所有数据集加载器，提供统一的创建和访问接口。
"""

from typing import Dict, Type, Optional, Any, List, Tuple, Union
from pathlib import Path

from config import ProjectConfig
# 使用延迟导入，避免在模块导入时就加载所有依赖（特别是DGL/OGB相关的警告）
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
    """
    统一数据加载器工厂
    
    管理所有数据集加载器，提供统一的创建和访问接口
    """
    
    _registry: Dict[str, Type[BaseDataLoader]] = {}
    _instances: Dict[str, BaseDataLoader] = {}
    
    @classmethod
    def register(cls, name: str, loader_class_or_factory) -> None:
        """
        注册数据加载器类或工厂函数
        
        Args:
            name: 数据集名称
            loader_class_or_factory: 数据加载器类或返回类的工厂函数
        """
        # 支持直接注册类或延迟导入的工厂函数
        cls._registry[name] = loader_class_or_factory
        logger.info(f"📝 注册数据加载器: {name} -> {getattr(loader_class_or_factory, '__name__', 'factory_function')}")
    
    @classmethod
    def create(cls, dataset_name: str, config: ProjectConfig, 
               target_property: Optional[str] = None,
               **kwargs) -> BaseDataLoader:
        """
        创建数据加载器实例
        
        Args:
            dataset_name: 数据集名称
            config: 项目配置
            target_property: 目标属性（对于多标签数据集）
            **kwargs: 传递给加载器的其他参数
            
        Returns:
            数据加载器实例
            
        Raises:
            ValueError: 当数据集名称未注册时
        """
        if dataset_name not in cls._registry:
            available_datasets = list(cls._registry.keys())
            raise ValueError(f"未知的数据集: {dataset_name}。已注册的数据集: {available_datasets}")
        
        # 检查是否已有实例（考虑目标属性）
        cache_key = f"{dataset_name}_{target_property}_{id(config)}"
        if cache_key in cls._instances:
            return cls._instances[cache_key]
        
        # 创建新实例 - 支持延迟导入
        loader_class_or_factory = cls._registry[dataset_name]
        
        # 如果是工厂函数，先调用获取实际的类
        if callable(loader_class_or_factory) and not hasattr(loader_class_or_factory, '__mro__'):
            loader_class = loader_class_or_factory()
        else:
            loader_class = loader_class_or_factory
            
        # 验证是否为BaseDataLoader的子类
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"数据加载器 {loader_class} 必须继承 BaseDataLoader")
            
        loader = loader_class(config, target_property=target_property, **kwargs)
        
        # 缓存实例
        cls._instances[cache_key] = loader
        
        logger.info(f"🔧 创建数据加载器: {dataset_name} (目标: {target_property})")
        return loader
    
    @classmethod
    def list_datasets(cls) -> Dict[str, Type[BaseDataLoader]]:
        """
        获取所有已注册的数据集
        
        Returns:
            Dict[str, Type[BaseDataLoader]]: 数据集名称到类的映射
        """
        return cls._registry.copy()
    
    @classmethod
    def get_loader_info(cls, dataset_name: str) -> Dict[str, Any]:
        """
        获取数据加载器信息
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            加载器信息字典
        """
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
        """清除所有缓存的数据加载器实例"""
        cls._instances.clear()
        logger.info("🗑️ 已清除所有数据加载器缓存")


def _lazy_import_loader(loader_name: str):
    """延迟导入数据加载器，避免在工厂初始化时导入所有依赖"""
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
    else:
        raise ValueError(f"Unknown loader: {loader_name}")

def _register_all_loaders():
    """注册所有数据加载器 - 使用字符串注册，延迟导入"""
    # 注册所有支持的加载器名称，实际类在使用时才导入
    loader_names = ["qm9", "qm9test", "zinc", "aqsol", "mnist", "mnist_raw", "colors3", "proteins", "synthetic", "mutagenicity", "coildel", "dblp", "dd", "twitter"]
    
    for name in loader_names:
        # 注册延迟导入函数而不是直接导入类
        UnifiedDataFactory.register(name, lambda n=name: _lazy_import_loader(n))
    
    logger.info("📝 数据加载器注册完成")


# 自动注册所有加载器
_register_all_loaders()


def get_dataloader(dataset_name: str, config: ProjectConfig, 
                   target_property: Optional[str] = None,
                   **kwargs) -> BaseDataLoader:
    """
    获取数据加载器实例
    
    Args:
        dataset_name: 数据集名称
        config: 项目配置
        target_property: 目标属性（对于多标签数据集）
        **kwargs: 传递给加载器的其他参数
        
    Returns:
        数据加载器实例
    """
    return UnifiedDataFactory.create(dataset_name, config, target_property, **kwargs)


def list_available_datasets() -> List[str]:
    """
    获取所有可用的数据集名称
    
    Returns:
        数据集名称列表
    """
    return list(UnifiedDataFactory.list_datasets().keys())


def get_dataset(dataset_name: str, config: ProjectConfig, 
                target_property: Optional[str] = None,
                **kwargs) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], 
                                  List[Any], List[Any], List[Any]]:
    """
    获取数据集（返回划分后的数据和标签）
    
    Args:
        dataset_name: 数据集名称
        config: 项目配置
        target_property: 目标属性（对于多标签数据集）
        **kwargs: 传递给加载器的其他参数
        
    Returns:
        Tuple: (训练数据, 验证数据, 测试数据, 训练标签, 验证标签, 测试标签)
    """
    loader = get_dataloader(dataset_name, config, target_property, **kwargs)
    return loader.load_data()


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    获取数据集信息
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        数据集信息字典
    """
    return UnifiedDataFactory.get_loader_info(dataset_name)

