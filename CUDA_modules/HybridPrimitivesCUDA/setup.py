from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hybrid_primitives',
    ext_modules=[
        CUDAExtension('hybrid_primitives_cuda', [
            'hybrid_primitives_cuda.cpp',
            'hybrid_primitives_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })