#
# This file tests that the CUDA kernels work as expected, compared with built-in PyTorch.
# A forward pass and backward pass is tested with multiple configurations for boundary conditions.
#

import torch
print("Loading CUDA kernels. May need to compile...")
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
print("Successfully loaded PeriodicPrimitives2D.")
from models.HybridPrimitives import HybridPrimitives
print("Successfully loaded HybridPrimitives.")
from time import time
from torch.profiler import profile, record_function, ProfilerActivity

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(7)

test_iters = 10
num_gaussians = 1
num_waves = 0
num_points = 1
num_dimensions = 2

hp = PeriodicPrimitives2D(gaussian_only=True)
hp.add_primitives_random(num_gaussians)

#hp = HybridPrimitives()
#hp.add_random_gaussians(num_gaussians)

x = torch.rand([num_points, num_dimensions], device="cuda", dtype=torch.float32)

def forward_memory_test():
    print(f"======================================================")
    print(f"=====================Memory test======================")
    print(f"======================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    
    print(f"Forward pass memory use (on top of parameters):")
    print(f"Static memory use:\t{starting_memory//1024**2}MB")


    '''
    try:
        _ = hp.forward_pytorch(x)
    except RuntimeError as e:
        print("Memory error - PyTorch exceeded the maximum GPU memory. Continuing test regardless.")
    ending_memory = torch.cuda.max_memory_allocated()
    pytorch_mem_use = ending_memory-starting_memory    
    print(f"PyTorch:\t\t{pytorch_mem_use//1024**2}MB")
    '''
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    _ = hp.forward(x)
    ending_memory = torch.cuda.max_memory_allocated()
    cuda_mem_use = ending_memory-starting_memory
    print(f"CUDA kernel:\t\t{cuda_mem_use//1024**2}MB")

    print(f"======================================================")

def backward_memory_test():
    print(f"======================================================")
    print(f"=====================Memory test======================")
    print(f"======================================================")

    print(f"Backward pass memory use (on top of parameters):")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    print(f"Static memory use:\t{starting_memory//1024**2}MB")
    try:
        out = hp.forward_pytorch(x)
        torch.abs(out).mean().backward()
        ending_memory = torch.cuda.max_memory_allocated()
        pytorch_mem_use = ending_memory-starting_memory
        print(f"PyTorch:\t\t{pytorch_mem_use//1024**2}MB")
    except RuntimeError as e:
        print("Memory error - PyTorch exceeded the maximum GPU memory. Continuing test regardless, stats may be incorrect.")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_memory = torch.cuda.max_memory_allocated()
    out = hp.forward(x)
    torch.abs(out).mean().backward()
    ending_memory = torch.cuda.max_memory_allocated()
    cuda_mem_use = ending_memory-starting_memory

    print(f"CUDA kernel:\t\t{cuda_mem_use//1024**2}MB")
    print(f"======================================================")

def forward_timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")
    print(f"Forward pass time:")

    torch.cuda.synchronize()
    t0 = time()
    for i in range(test_iters):
        _ = hp.forward(x)
    torch.cuda.synchronize()
    time_cuda = time() - t0 + 1e-12
    print(f"CUDA kernel:\t\t{time_cuda/test_iters:0.09f} sec. per pass \t {test_iters/time_cuda} FPS")

    '''
    torch.cuda.synchronize()
    t0 = time()
    for i in range(test_iters):
        try:
            _ = hp.forward_pytorch(x)
        except RuntimeError as e:
            print("Memory error - PyTorch exceeded the maximum GPU memory. Continuing test regardless.")
            break
    torch.cuda.synchronize()
    time_pytorch = time() - t0 + 1e-12

    print(f"PyTorch:\t\t{time_pytorch/test_iters:0.09f} sec. per pass \t {test_iters/time_pytorch} FPS")
    print(f"CUDA speedup:\t\t{time_pytorch/time_cuda:0.02f}x")
    '''

    print(f"======================================================")

def backward_timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")
    print(f"Backward pass time:")

    t0 = time()
    torch.cuda.synchronize()
    for i in range(test_iters):
        try:
            out = hp.forward_pytorch(x)
        except RuntimeError as e:
            print("Memory error - PyTorch exceeded the maximum GPU memory. Continuing test regardless.")
            break
        out.mean().backward()
    torch.cuda.synchronize()
    time_pytorch = time() - t0
    print(f"PyTorch:\t\t{time_pytorch/test_iters:0.09f} sec. per pass")

    torch.cuda.synchronize()
    t0 = time()
    for i in range(test_iters):
        out = hp.forward(x)
        out.mean().backward()
    torch.cuda.synchronize()
    time_cuda = time() - t0

    print(f"CUDA kernel:\t\t{time_cuda/test_iters:0.09f} sec. per pass  \t {test_iters/time_cuda} FPS")
    print(f"CUDA speedup:\t\t{time_pytorch/(time_cuda+1e-9):0.02f}x")
    print(f"======================================================")

def inference_timing_test():
    print(f"======================================================")
    print(f"=====================Timing test======================")
    print(f"======================================================")
    with torch.no_grad():
        t0 = time()
        torch.cuda.synchronize()
        for i in range(test_iters):
            try:
                _ = hp.forward_pytorch(x)
            except RuntimeError as e:
                print("Memory error - PyTorch exceeded the maximum GPU memory. Continuing test regardless.")
                break
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
    try:
        out_pytorch = hp.forward_pytorch(x)
    except RuntimeError as e:
        print("Memory error - PyTorch exceeded the maximum GPU memory. Cant assess error.")
        raise e
    out_cuda = hp.forward(x)
    print(out_pytorch)
    print(out_cuda)
    error = torch.abs(out_pytorch-out_cuda).flatten()
    mse = error.mean()
    max_error = error.max()
    print(f"Mean absolute error:\t\t{mse}")
    print(f"Max absolute error: \t\t{max_error}")
    print(f"======================================================")

def backward_error_test():
    print(f"======================================================")
    print(f"===================Backward test======================")
    print(f"======================================================")
    try:
        out_pytorch = hp.forward_pytorch(x)
    except RuntimeError as e:
        print("Memory error - PyTorch exceeded the maximum GPU memory. Ending test.")
        return

    torch.set_printoptions(threshold=10)
    (out_pytorch.abs()).mean().backward()
    groups_pytorch = []
    for group in hp.optimizer.param_groups:
        if(group["params"][0].numel() > 0 and group["params"][0].grad is not None):
            grads = group['params'][0].grad.clone().detach()
            print(grads)
            groups_pytorch.append(grads)

    hp.zero_grad()
    hp.optimizer.zero_grad()
    
    out_cuda = hp.forward(x)
    (out_cuda.abs()).mean().backward()
    groups_cuda = []
    for group in hp.optimizer.param_groups:
        if(group["params"][0].numel() > 0 and group["params"][0].grad is not None):
            grads = group['params'][0].grad.clone().detach()
            print(grads)
            groups_cuda.append(grads)

    assert len(groups_cuda) == len(groups_pytorch), "Parameter groups do not match between PyTorch and CUDA"
    err = 0
    for i in range(len(groups_pytorch)):
        assert groups_pytorch[i].shape == groups_cuda[i].shape, f"Parameter group sizes dont match at index {i}"
        if(groups_pytorch[i].numel() > 0):
            this_err = torch.abs(groups_pytorch[i]-groups_cuda[i]).sum()
            print(this_err)
            err += this_err
    print(f"Total gradient error: {err}")
        
    print(f"======================================================")

def profiler_test():
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        y = hp.forward(x)
        torch.abs(y).mean().backward()
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=5))


forward_error_test()
backward_error_test()

#forward_memory_test()
#backward_memory_test()

forward_timing_test()
#inference_timing_test()
#backward_timing_test()
#profiler_test()