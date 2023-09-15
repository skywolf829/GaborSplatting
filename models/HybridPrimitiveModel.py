import torch
from torch.nn.parameter import Parameter
from models.LombScargle import MyLombScargleModel
from models.GaussianSplatting2D import GaussianSplatting2D
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
from models.SupportedPeriodicPrimitives2D import SupportedPeriodicPrimitives2D
import matplotlib.pyplot as plt

class HybridPrimitiveModel(torch.nn.Module):

    def __init__(self, n_waves, n_gaussians, device="cuda"):
        super().__init__()
        # Parameters
        self.gaussians = GaussianSplatting2D(n_gaussians, device)
        self.periodic_primitives = SupportedPeriodicPrimitives2D(n_waves, device)

    def train_model(self, x, y, im_shape):
        self.periodic_primitives.train_model(x, y, im_shape)
        with torch.no_grad():
            res = y - self.periodic_primitives(x)
        self.gaussians.train_model(x, res, im_shape)
        model_out = self(x)
        plt.scatter(x[:,0].detach().cpu().numpy(), x[:,1].detach().cpu().numpy(),
                    c=model_out.detach().cpu().numpy())
        plt.imsave("./output/HybridModelOut.png")

    def forward(self, x):        
        periodic_part = self.periodic_primitives(x)
        gaussian_part = self.gaussians(x)
        return periodic_part + gaussian_part

