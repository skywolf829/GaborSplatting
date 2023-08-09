import torch
from torch.nn.parameter import Parameter
from utils.math_utils import inv2x2

class GaussianSplatting2D(torch.nn.Module):

    def __init__(self, n_gaussians, device="cuda"):
        super().__init__()

        # Parameters
        #self.colors = Parameter(torch.rand([n_gaussians, 1], dtype=torch.float32, device=device))
        self.alphas = Parameter(((1/n_gaussians)**0.5)*torch.rand([n_gaussians, 1], dtype=torch.float32, device=device))
        self.means = Parameter(torch.rand([n_gaussians, 2], dtype=torch.float32, device=device))
        # RSSR
        self.rotations = Parameter(torch.pi*torch.rand([n_gaussians, 1], dtype=torch.float32, device=device))
        self.scales = Parameter(0.1*torch.rand([n_gaussians, 2], dtype=torch.float32, device=device))


    def create_optimizer(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001, 
            betas=[0.9, 0.99])
        return optim

    def create_rotation_matrices(self):
        return torch.stack([[torch.cos(self.rotations), -torch.sin(self.rotations)],
                           [torch.sin(self.rotations), torch.cos(self.rotations)]], dim=-1).reshape(-1, 2, 2)

    def create_RS(self):
        return torch.stack([self.scales[:,0:1]*torch.cos(self.rotations), -self.scales[:,1:2]*torch.sin(self.rotations),
                            self.scales[:,0:1]*torch.sin(self.rotations), self.scales[:,1:2]*torch.cos(self.rotations)], 
                            dim=-1).reshape(-1, 2, 2)

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        l1 = torch.nn.functional.l1_loss(model_out,y)
        return l1, model_out
    
    def forward(self, x):
        # x is [N, 2]
        rel_x = x[:,None,:] - self.means[:,...] # [N, n_gaussians, 2]
        RS = self.create_RS() # [n_gaussians, 2, 2]
        RSSR = RS @ RS.mT
        cov = inv2x2(RSSR)
        # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
        transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
        # [N, n_gaussians, 1, 1]
        transformed_x = torch.exp(-transformed_x[:,:,0]/2) # [N, n_gaussians]
        vals = self.alphas*transformed_x # [N, n_gaussians, channels]
        vals = torch.sum(vals, dim=1)
        return vals # [N, channels]

