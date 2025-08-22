"""
TokenizerGraph项目
================

图结构序列化与压缩的研究项目
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置项目日志
from utils.logger import setup_logger, get_logger

# 初始化项目级别的logger
PROJECT_LOGGER = setup_logger('tokenizerGraph', level='INFO')

__version__ = '1.0.0'
__author__ = 'TokenizerGraph Team'

# 延迟导入，避免在测试收集阶段不必要的重依赖加载
# from . import data
# from . import algorithms
from . import utils

__all__ = [
    "data",
    "algorithms",
    "utils"
] 