"""
统一模型工厂
============

create_universal_model - 统一的模型创建接口
支持预训练(MLM)和微调(分类/回归)的无缝切换
"""

from __future__ import annotations
from typing import Tuple, Dict

from src.models.universal_model import UniversalModel
from src.models.unified_encoder import create_encoder
from src.training.task_handler import TaskHandler, create_task_handler
from src.utils.logger import get_logger

# 创建模块级logger
logger = get_logger(__name__)


