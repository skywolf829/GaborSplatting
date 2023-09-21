from typing import Optional
import torch
from torch.nn.parameter import Parameter
import time
from models.LombScargle2D import LombScargle2D
from models.GaussianLombScargle import GaussianLombScargleModel
from tqdm import tqdm
from utils.data_utils import to_img
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import psnr
import tinycudann as tcnn
torch.backends.cuda.matmul.allow_tf32 = True

class Siren(torch.nn.Module):

    def __init__(self, n_primitives, n_channels=1, device="cuda"):
        super().__init__()
        # Parameters
        
        self.model = tcnn.NetworkWithInputEncoding(
            2, n_channels,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 12
                },
            network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "Sine",
                    "output_activation": "None",
                    "n_neurons": 32,
                    "n_hidden_layers": 2,                                
                }
            )

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-15)
        return optimizer
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:           
            total += group['params'][0].numel()
        return total

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        final_loss = mse 
        losses = {
            "final_loss": final_loss,
            "mse": mse
        }
        return losses, model_out

    def forward(self, x: torch.Tensor):
        return self.model(x).float()

