# Example file for how to load the cuda extension with JIT

from torch.utils.cpp_extension import load
import os
this_folder_path = __file__.split('jit.py')[0]
print(this_folder_path)
periodic_primitives = load(name='periodic_primitives', 
        sources=[os.path.join(this_folder_path,'periodic_primitives2DRGB_cuda.cpp'), 
        os.path.join(this_folder_path,'periodic_primitives2DRGB_cuda_kernel.cu')])