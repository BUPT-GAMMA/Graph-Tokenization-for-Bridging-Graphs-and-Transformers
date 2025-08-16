"""
数据层模块 - 统一的图数据集接口
============================

提供统一的图数据集加载和处理接口，支持多种图数据集。
"""

# from .qm9_loader import QM9Loader
# from .qm9_loader import QM9Loader  # 避免在导入数据层时强制导入 DGL 依赖的数据集

# 为避免在导入本包时触发 DGL 的重依赖，仅在需要时由调用方延迟导入
# from .single_graph_loader import SingleGraphLoader, load_single_graph_dataset

# 新的统一数据加载器工厂 - 推荐使用
from .unified_data_factory import (
    UnifiedDataFactory,
    get_dataloader,
    get_dataset,
    list_available_datasets,
    get_dataset_info
)

# 统一数据接口
from .unified_data_interface import UnifiedDataInterface

# 预处理数据加载接口 (即将废弃，请使用 UnifiedDataInterface)
# 暂时保留以避免立即破坏现有代码

# 导出公共接口
__all__ = [
    # 新的统一数据加载器工厂 - 推荐使用
    'UnifiedDataFactory',
    'get_dataloader',
    'get_dataset',
    'list_available_datasets',
    'get_dataset_info',
    
    # 统一数据接口 - 推荐使用
    'UnifiedDataInterface',
]
