#
# This file tests that the CUDA kernels work as expected, compared with built-in PyTorch.
# A forward pass and backward pass is tested with multiple configurations for boundary conditions.
#

import torch
print("Loading HybridPrimitives CUDA kernel. May need to compile...")
from models.HybridPrimitives import HybridPrimitives
print("Successfully loaded HybridPrimitives.")
from time import time

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(7)

test_iters = 10
num_gaussians = 1000
num_waves = 100
num_points = 100000
num_dimensions = 2

hp = HybridPrimitives()
hp.add_gaussians(num_gaussians)
hp.add_waves(num_waves)

x = torch.rand([num_points, num_dimensions], device="cuda", dtype=torch.float32)

def forward_memory_test():
    print(f"======================================================")
    print(f"=====================Memory test======================")
    print(f"======================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
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

def backward_memory_test():
    print(f"======================================================")
    print(f"=====================Memory test======================")
    print(f"======================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    out = hp.forward_pytorch(x)
    torch.abs(out).mean().backward()
    ending_memory = torch.cuda.max_memory_allocated()
    pytorch_mem_use = ending_memory-starting_memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    out = hp.forward(x)
    torch.abs(out).mean().backward()
    ending_memory = torch.cuda.max_memory_allocated()
    cuda_mem_use = ending_memory-starting_memory

    print(f"Backward pass memory use (on top of parameters):")
    print(f"Static memory use:\t{starting_memory//1024**2}MB")
    print(f"PyTorch:\t\t{pytorch_mem_use//1024**2}MB")
    print(f"CUDA kernel:\t\t{cuda_mem_use//1024**2}MB")
    print(f"======================================================")

def forward_timing_test():
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

def backward_timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")

    t0 = time()
    torch.cuda.synchronize()
    for i in range(test_iters):
        out = hp.forward_pytorch(x)
        out.mean().backward()
    torch.cuda.synchronize()
    time_pytorch = time() - t0

    torch.cuda.synchronize()
    t0 = time()
    for i in range(test_iters):
        out = hp.forward(x)
        out.mean().backward()
    torch.cuda.synchronize()
    time_cuda = time() - t0

    print(f"Backward pass time:")
    print(f"PyTorch:\t\t{time_pytorch/1000:0.09f} sec.")
    print(f"CUDA kernel:\t\t{time_cuda/1000:0.09f} sec.")
    print(f"CUDA speedup:\t\t{time_pytorch/time_cuda:0.02f}x")
    print(f"======================================================")

def inference_timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")
    with torch.no_grad():
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

        print(f"Inference pass time:")
        print(f"PyTorch:\t\t{time_pytorch/1000:0.09f} sec.")
        print(f"CUDA kernel:\t\t{time_cuda/1000:0.09f} sec.")
        print(f"CUDA speedup:\t\t{time_pytorch/time_cuda:0.02f}x")
        print(f"======================================================")

def forward_error_test():
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
    assert max_error < 1e-8, f"Error above 1e-8"
    print(f"======================================================")

def backward_error_test():
    print(f"======================================================")
    print(f"===================Backward test======================")
    print(f"======================================================")

    out_pytorch = hp.forward_pytorch(x)
    out_cuda = hp.forward(x)

    (out_pytorch.abs()).mean().backward()
    groups_pytorch = []
    for group in hp.optimizer.param_groups:
        groups_pytorch.append(group['params'][0].clone().detach())

    hp.zero_grad()

    (out_cuda.abs()).mean().backward()
    groups_cuda = []
    for group in hp.optimizer.param_groups:
        groups_cuda.append(group['params'][0].clone().detach())

    assert len(groups_cuda) == len(groups_pytorch), "Parameter groups do not match between PyTorch and CUDA"
    err = 0
    for i in range(len(groups_pytorch)):
        assert groups_pytorch[i].shape == groups_cuda[i].shape, f"Parameter group sizes dont match at index {i}"
        if(groups_pytorch[i].numel() > 0):
            this_err = ((groups_pytorch[i]-groups_cuda[i])**2).mean()**0.5
            err += this_err
    print(f"Total gradient error: {err}")
        
    print(f"======================================================")


forward_error_test()
backward_error_test()

forward_memory_test()
backward_memory_test()

forward_timing_test()
inference_timing_test()
backward_timing_test()