# Example file for how to load the cuda extension with JIT

from torch.utils.cpp_extension import load
import os

periodic_primitives = load(name='periodic_primitives', 
                    sources=[os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),'periodic_primitives2DRGB_cuda.cpp'), 
                             os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),'periodic_primitives2DRGB_cuda_kernel.cu')])