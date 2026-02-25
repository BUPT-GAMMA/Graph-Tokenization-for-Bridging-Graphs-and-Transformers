"""Compression algorithms (BPE).
压缩算法（BPE）。"""

from .main_bpe import StandardBPECompressor  # reference implementation (Python)
from .bpe_engine import BPEEngine  # unified train/encode entry (C++ train + C++/Python encode)

# Optional backend (C++ encode only)
try:
    from .cpp_bpe_backend import CppBPEBackend  # noqa: F401
    _cpp_available = True
except Exception:
    _cpp_available = False

__all__ = ['StandardBPECompressor', 'BPEEngine']
if _cpp_available:
    __all__.append('CppBPEBackend')