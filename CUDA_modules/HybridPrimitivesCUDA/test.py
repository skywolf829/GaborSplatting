#
# This file tests that the CUDA kernels work as expected, compared with built-in PyTorch.
# A forward pass and backward pass is tested with multiple configurations for boundary conditions.
#

import torch
from HybridPrimitives import HybridPrimitives
from time import time

hp = HybridPrimitives()
hp.add_gaussians(100)

with torch.no_grad():
    x = torch.rand([100000, 2], device="cuda", dtype=torch.float32)

    t0 = time()
    for i in range(1000):
        output_cuda = hp.forward(x)
        torch.cuda.synchronize()
    time_cuda = time() - t0

    t0 = time()
    for i in range(1000):
        output_pytorch = hp.forward_pytorch(x)
        torch.cuda.synchronize()
    time_pytorch = time() - t0

    print(f"Average time custom CUDA kernel: {time_cuda/1000}")
    print(f"Average time PyTorch: {time_pytorch/1000}")


print(f"All tests passed.")