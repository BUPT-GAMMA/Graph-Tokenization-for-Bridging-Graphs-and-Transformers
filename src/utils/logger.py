"""
统一日志配置模块
==============

为整个项目提供统一的日志配置和管理
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# 默认日志格式
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s - %(name)s - %(message)s'

class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 获取原始格式化的消息
        formatted = super().format(record)
        
        # 添加颜色
        if hasattr(record, 'levelname') and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"
        
        return formatted

def setup_logger(
    name: str,
    level: str = 'INFO',
    format_str: str = DEFAULT_FORMAT,
    console_output: bool = True,
    file_output: Optional[str] = None,
    colored: bool = True
) -> logging.Logger:
    """
    设置并返回配置好的logger
    
    Args:
        name: logger名称
        level: 日志级别
        format_str: 日志格式
        console_output: 是否输出到控制台
        file_output: 文件输出路径
        colored: 是否使用彩色输出
    
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建格式化器
    if colored and console_output:
        formatter = ColoredFormatter(format_str)
    else:
        formatter = logging.Formatter(format_str)
    
    # 控制台输出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file_output:
        file_path = Path(file_output)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# 预设的logger实例
def get_logger(name: str = None, level: str = 'INFO') -> logging.Logger:
    """
    获取logger实例的快捷方法
    
    Args:
        name: logger名称，如果为None则使用调用模块的名称
        level: 日志级别
    
    Returns:
        logger实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return setup_logger(name, level=level)

# 项目级别的logger
PROJECT_LOGGER = get_logger('tokenizerGraph')
DATA_LOGGER = get_logger('tokenizerGraph.data')
ALGORITHM_LOGGER = get_logger('tokenizerGraph.algorithm')
COMPRESSION_LOGGER = get_logger('tokenizerGraph.compression') 