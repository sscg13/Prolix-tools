import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Detect compiler environment
if sys.platform == 'win32':
    # Microsoft Visual C++ (MSVC) flags
    compile_args = ['/O2', '/arch:AVX2', '/openmp']
else:
    # GCC / Clang flags
    compile_args = ['-O3', '-march=native', '-fopenmp']

setup(
    name='cpp_tuner',
    ext_modules=[
        CppExtension(
            name='cpp_tuner', 
            sources=['tuner.cpp'],
            extra_compile_args=compile_args
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)