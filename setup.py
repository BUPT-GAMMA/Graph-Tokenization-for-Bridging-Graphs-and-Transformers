from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os

try:
    import pybind11
except Exception as e:  # noqa: E722
    print("[WARN] pybind11 未安装，无法构建 C++ 扩展。请先 pip install pybind11。", file=sys.stderr)
    raise


class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()


ext_modules = [
    Extension(
        'src.algorithms.compression._cpp_bpe',
        sources=['src/algorithms/compression/_cpp_bpe.cpp'],
        include_dirs=[get_pybind_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-march=native', '-flto', '-DNDEBUG']
    ),
]


setup(
    name='tokenizerGraph_cpp_ext',
    version='0.0.1',
    author='tokenizerGraph',
    description='C++ backend for BPE encode',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)


