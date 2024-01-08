from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='PeriodicPrimitives',
    ext_modules=[
        CUDAExtension('PeriodicPrimitives', [
            'periodic_primitives2DRGB_cuda.cpp',
            'periodic_primitives2DRGB_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })