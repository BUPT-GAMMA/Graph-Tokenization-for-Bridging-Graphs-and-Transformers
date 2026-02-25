"""
Unified logging module.
统一日志模块。

Centralized logging configuration and management for the project.
项目的集中式日志配置和管理。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Log level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Default log formats
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s - %(name)s - %(message)s'

class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # cyan
        'INFO': '\033[32m',      # green
        'WARNING': '\033[33m',   # yellow
        'ERROR': '\033[31m',     # red
        'CRITICAL': '\033[35m',  # magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Get base formatted message
        formatted = super().format(record)
        
        # Add color
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
    Set up and return a configured logger.
    
    Args:
        name: Logger name
        level: Log level
        format_str: Log format string
        console_output: Whether to output to console
        file_output: File output path
        colored: Whether to use colored output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    if colored and console_output:
        formatter = ColoredFormatter(format_str)
    else:
        formatter = logging.Formatter(format_str)
    
    # Console output
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File output
    if file_output:
        file_path = Path(file_output)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Preset logger instances
def get_logger(name: str = None, level: str = 'INFO') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name; uses caller module name if None
        level: Log level
    
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return setup_logger(name, level=level)

# Project-level loggers
PROJECT_LOGGER = get_logger('tokenizerGraph')
DATA_LOGGER = get_logger('tokenizerGraph.data')
ALGORITHM_LOGGER = get_logger('tokenizerGraph.algorithm')
COMPRESSION_LOGGER = get_logger('tokenizerGraph.compression') 