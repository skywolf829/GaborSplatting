from torch.utils.cpp_extension import load

hybrid_primitives = load(name='hybrid_primitives', 
                    sources=['hybrid_primitives_cuda.cpp', 
                             'hybrid_primitives_cuda_kernel.cu'],
                    verbose=True)
