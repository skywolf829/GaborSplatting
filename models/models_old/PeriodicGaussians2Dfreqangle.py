from typing import Optional
import torch
from torch.nn.parameter import Parameter
import time
from models.LombScargle2D import LombScargle2D
from models.LombScargle2Danglefreq import LombScargle2Danglefreq
from models.GaussianLombScargle import GaussianLombScargleModel
from tqdm import tqdm
from utils.data_utils import to_img
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import psnr
import tinycudann as tcnn
torch.backends.cuda.matmul.allow_tf32 = True

class PeriodicGaussians2Dfreqangle(torch.nn.Module):

    def __init__(self, n_primitives, n_channels=1, device="cuda"):
        super().__init__()
        # Parameters
        self.n_primitives = n_primitives
        self.device = device
        self.n_channels = n_channels

        self.gaussian_means = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.gaussian_mats = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.subgaussian_frequency = Parameter(torch.empty(0, device=device, dtype=torch.float32))    
        self.subgaussian_coefficients = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.subgaussian_rotation = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        
        self.colors = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.avg_color = torch.zeros([3], device=device, dtype=torch.float32)
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_means], 'lr': 0.001, "name": "gaussian_means"},
            {'params': [self.gaussian_mats], 'lr': 0.001, "name": "gaussian_mats"},

            {'params': [self.subgaussian_frequency], 'lr': 0.0001, "name": "subgaussian_frequency"},
            {'params': [self.subgaussian_coefficients], 'lr': 0.001, "name": "subgaussian_coefficients"},
            {'params': [self.subgaussian_rotation], 'lr': 0.001, "name": "subgaussian_rotation"},
            
            {'params': [self.colors], 'lr': 0.00001, "name": "colors"}
        ]

        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:           
            total += group['params'][0].numel()
        return total

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(group['name'] in tensors_dict):
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_tensors_from_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(group['params'][0].shape[0] == mask.shape[0]):
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_primitives(self, min_contribution:int=1./255.):
        if(self.gaussian_means.shape[0] == 0):
            return
        mask = (torch.linalg.norm(self.colors,dim=-1) * torch.linalg.norm(self.subgaussian_coefficients,dim=-1)) > min_contribution
        if(len(mask.shape)>0):
            to_remove = mask.shape[0]-mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                updated_params = self.prune_tensors_from_optimizer(mask)

                self.colors = updated_params['colors']
                self.gaussian_means = updated_params['gaussian_means']
                self.gaussian_mats = updated_params['gaussian_mats']
                self.subgaussian_coefficients = updated_params['subgaussian_coefficients']
                self.subgaussian_frequency = updated_params['subgaussian_frequency']
                self.subgaussian_rotation = updated_params['subgaussian_rotation']

    def add_primitives(self, x, y, n_freqs=256, n_angles=180, 
                       freq_decay = 1.0, min_influence=1./255.,
                       num_to_add=1):
        #plt.scatter(x.cpu().numpy()[:,1], x.cpu().numpy()[:,0],c=y.cpu().numpy())
        #plt.show()
        ls_model = LombScargle2Danglefreq(x, y, n_terms=1, device=self.device)
        # Randomize set of wavelengths and freqs to fit in correct range
        freqs = 0.1+torch.sort(freq_decay*64*torch.rand([n_freqs], device=self.device, dtype=torch.float32)).values
        angles = torch.sort(torch.pi*0.5*torch.rand([n_angles], device=self.device, dtype=torch.float32)).values
        
        # Fit model and find peaks
        ls_model.fit(freqs, angles)            
        self.data_periodogram = ls_model.get_power()
        n_extracted_peaks = ls_model.find_peaks(top_n=num_to_add, 
                                                min_influence=min_influence)
        self.ls_plot = ls_model.plot_power(return_img=True)
        self.ls_power = ls_model.power
        all_colors = ls_model.get_PCA_color()[None,:].repeat(n_extracted_peaks, 1) / n_extracted_peaks

        all_frequencies = ls_model.get_peak_freqs()
        all_angles = all_frequencies[:,0]
        all_frequencies = all_frequencies[:,1]
        coeffs = ls_model.get_peak_coeffs()

        means, vars = ls_model.get_peak_placement(torch.arange(0,n_extracted_peaks,1,dtype=torch.long,device=self.device))
        #means = means*0 + 0.5  
        mats = torch.eye(2, device=self.device, dtype=torch.float32)[None,...].repeat(n_extracted_peaks, 1, 1)
        mats[:,0,0] = 1 / vars[:,0]
        mats[:,1,1] = 1 / vars[:,1]

        tensor_dict = {
            "gaussian_means": means, 
            "gaussian_mats": mats,
            "subgaussian_frequency": all_frequencies,
            "subgaussian_rotation": all_angles,
            "subgaussian_coefficients": coeffs,
            "colors": all_colors
        }
        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.gaussian_means = updated_params['gaussian_means']
        self.gaussian_mats = updated_params['gaussian_mats']
        self.subgaussian_frequency = updated_params['subgaussian_frequency']
        self.subgaussian_rotation = updated_params['subgaussian_rotation']
        self.subgaussian_coefficients = updated_params['subgaussian_coefficients']
        self.colors = updated_params['colors']

        return n_extracted_peaks
        
    def vis_primitives(self, x, res=[256, 256], power=10):
        
        with torch.no_grad():
            xmin = x.min(dim=0).values
            xmax = x.max(dim=0).values
            g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            x = g
            # x is [N, 2]
            # gaussian coeffs
            vals = self.forward(g, sum_together=False)

            im_per_row = 4
            nwaves = max(self.gaussian_means.shape[0], self.n_primitives)
            n_cols = min(nwaves, im_per_row)
            n_rows = 1+(nwaves//im_per_row)
            im = torch.zeros([n_rows*res[0], n_cols*res[1], self.n_channels])
            
            row_spot = 0
            col_spot = 0
            
            for i in range(vals.shape[1]):

                im[row_spot*res[0]:(row_spot+1)*res[0],
                   col_spot*res[1]:(col_spot+1)*res[1]] = vals[:,i].detach().cpu().reshape(res+[self.n_channels])

                if((i+1) % im_per_row == 0):
                    row_spot += 1
                    col_spot = 0
                else:
                    col_spot += 1
        return im

    def create_RS(self):
        #return torch.stack([torch.exp(self.scales[:,0:1])*torch.cos(self.gaussian_rotations), 
        #                              torch.exp(self.scales[:,1:2])*-torch.sin(self.gaussian_rotations),
        #                    torch.exp(self.scales[:,0:1])*torch.sin(self.gaussian_rotations), 
        #                              torch.exp(self.scales[:,1:2])*torch.cos(self.gaussian_rotations)], 
        #                    dim=-1).reshape(-1, 2, 2)
        return self.gaussian_mats

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        shear_loss = 0.01*((self.gaussian_mats[:,1,0]+self.gaussian_mats[:,0,1])**2).mean()
        final_loss = mse #+ shear_loss+ decay_loss
        losses = {
            "final_loss": final_loss,
            "mse": mse,
            "shear_loss": shear_loss
        }
        return losses, model_out

    def forward(self, x: torch.Tensor, sum_together: Optional[bool]=True):
        # x is [N, 2]
        # gaussian coeffs
        rel_x = x[:,None,:] - self.gaussian_means[None,...] # [N, n_gaussians, 2]
        RS = self.create_RS() # [n_gaussians, 2, 2]
        cov = RS #@ RS.mT
        #cov = inv2x2(cov)
        # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
        #transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
        transformed_x = ((rel_x[...,None].mT @ cov[None,...])**10).sum(dim=-1, keepdim=True)
        #transformed_x = rel_x[...,None].mT @ cov[None,...] @ cov[None,...].mT @ rel_x[...,None]
        # [N, n_gaussians, 1, 1]
        gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
        
        # periodic gaussian parts
        # Fitted sinusoidal wave
        # [N, channels]
        
        x_r = x[:,0:1]*torch.cos(self.subgaussian_rotation[None,:]) + \
                x[:,1:2]*torch.sin(self.subgaussian_rotation[None,:])
        x_r = 2*torch.pi*x_r*self.subgaussian_frequency[None,:]
        vals = self.subgaussian_coefficients[None,:,0,0]*torch.cos(x_r) + \
                self.subgaussian_coefficients[None,:,0,1]*torch.sin(x_r) + \
                self.subgaussian_coefficients[None,:,0,2]
        
        # [N, n_primitives, 1]
        vals = gauss_vals*vals[...,None]*self.colors[None,...]
        if(sum_together):
            vals = vals.sum(dim=1)
        return vals 

