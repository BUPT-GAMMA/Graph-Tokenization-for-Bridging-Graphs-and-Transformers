"""
压缩算法模块

包含BPE压缩相关的功能
"""

from .main_bpe import StandardBPECompressor  # 参考实现（Python）
from .bpe_engine import BPEEngine  # 统一训练/编码入口（C++ 训练 + C++/Python 编码）

# 可选后端（仅暴露 C++ 编码后端，训练走 Python/Numba 引擎）
try:
    from .cpp_bpe_backend import CppBPEBackend  # noqa: F401
    _cpp_available = True
except Exception:
    _cpp_available = False

__all__ = ['StandardBPECompressor', 'BPEEngine']
if _cpp_available:
    __all__.append('CppBPEBackend')