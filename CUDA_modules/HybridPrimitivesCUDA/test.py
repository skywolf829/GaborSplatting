#
# This file tests that the CUDA kernels work as expected, compared with built-in PyTorch.
# A forward pass and backward pass is tested with multiple configurations for boundary conditions.
#

import torch
print("Loading HybridPrimitives CUDA kernel. May need to compile...")
from HybridPrimitives import HybridPrimitives
print("Successfully loaded HybridPrimitives.")
from time import time

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


test_iters = 10
num_gaussians = 10000
num_waves = 500
num_points = 500
num_dimensions = 2

hp = HybridPrimitives()
hp.add_gaussians(num_gaussians)
hp.add_waves(num_waves)

x = torch.rand([num_points, num_dimensions], device="cuda", dtype=torch.float32)

def memory_test():
    print(f"======================================================")
    print(f"=====================Memory test======================")
    print(f"======================================================")

    starting_memory = torch.cuda.max_memory_allocated()
    _ = hp.forward_pytorch(x)
    ending_memory = torch.cuda.max_memory_allocated()
    pytorch_mem_use = ending_memory-starting_memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    _ = hp.forward(x)
    ending_memory = torch.cuda.max_memory_allocated()
    cuda_mem_use = ending_memory-starting_memory

    print(f"Forward pass memory use (on top of parameters):")
    print(f"Static memory use:\t{starting_memory//1024**2}MB")
    print(f"PyTorch:\t\t{pytorch_mem_use//1024**2}MB")
    print(f"CUDA kernel:\t\t{cuda_mem_use//1024**2}MB")
    print(f"======================================================")

def timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")

    t0 = time()
    torch.cuda.synchronize()
    for i in range(test_iters):
        _ = hp.forward_pytorch(x)
    torch.cuda.synchronize()
    time_pytorch = time() - t0

    torch.cuda.synchronize()
    t0 = time()
    for i in range(test_iters):
        _ = hp.forward(x)
    torch.cuda.synchronize()
    time_cuda = time() - t0

    print(f"Forward pass time:")
    print(f"PyTorch:\t\t{time_pytorch/1000:0.09f} sec.")
    print(f"CUDA kernel:\t\t{time_cuda/1000:0.09f} sec.")
    print(f"CUDA speedup:\t\t{time_pytorch/time_cuda:0.02f}x")
    print(f"======================================================")

def forward_test():
    print(f"======================================================")
    print(f"====================Forward test======================")
    print(f"======================================================")

    out_pytorch = hp.forward_pytorch(x)
    out_cuda = hp.forward(x)
    error = ((out_pytorch-out_cuda)**2).flatten()
    mse = error.mean()
    max_error = error.max()
    print(f"MSE:\t\t\t\t{mse}")
    print(f"Max squared error \t\t{max_error}")

    print(f"======================================================")

def backward_test():
    print(f"======================================================")
    print(f"===================Backward test======================")
    print(f"======================================================")

    out_pytorch = hp.forward_pytorch(x)
    out_cuda = hp.forward(x)
    (out_pytorch**2).mean().backward()
    print(hp.gaussian_colors.grad[0])
    hp.zero_grad()
    (out_cuda**2).mean().backward()
    print(hp.gaussian_colors.grad[0])
        
    print(f"======================================================")

with torch.no_grad():    
    memory_test()
    timing_test()
    forward_test()
backward_test()