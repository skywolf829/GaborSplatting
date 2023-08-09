import torch

def inv2x2(x):
    # [N, 2, 2]
    coeff = (x[:,0,0]*x[:,1,1] - x[:,0,1]*x[:,1,0])
    mat = torch.stack([
        x[:,1,1], -x[:,0,1], -x[:,1,0], x[:,0,0]
    ], dim=-1).reshape(-1, 2, 2)
    return mat / (coeff[:,None,None]+1e-6)