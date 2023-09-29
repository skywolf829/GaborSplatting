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

class HybridPrimitiveModel(torch.nn.Module):

    def __init__(self, n_primitives, n_channels=1, device="cuda"):
        super().__init__()

        self.n_primitives = n_primitives
        self.device = device
        self.n_channels = n_channels

        self.gaussian_colors = Parameter(torch.empty(0, device=device, dtype=torch.float32)) 
        self.gaussian_means = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.gaussian_mats = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        
        self.wave_colors = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.wave_support_means = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.wave_support_mats = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.wave_frequencies = Parameter(torch.empty(0, device=device, dtype=torch.float32))    
        self.wave_coefficients = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_colors], 'lr': 0.01, "name": "gaussian_colors"},
            {'params': [self.gaussian_means], 'lr': 0.01, "name": "gaussian_means"},
            {'params': [self.gaussian_mats], 'lr': 0.01, "name": "gaussian_mats"},

            {'params': [self.wave_colors], 'lr': 0.001, "name": "wave_colors"},
            {'params': [self.wave_support_means], 'lr': 0.001, "name": "wave_support_means"},
            {'params': [self.wave_support_mats], 'lr': 0.001, "name": "wave_support_mats"},
            {'params': [self.wave_frequencies], 'lr': 0.001, "name": "wave_frequencies"},
            {'params': [self.wave_coefficients], 'lr': 0.001, "name": "wave_coefficients"},
            
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
    
    def prune_tensors_from_optimizer(self, mask, type="waves"):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(type in group['name'] and group['params'][0].shape[0] == mask.shape[0]):
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
        gaussians_mask = torch.linalg.norm(self.gaussian_colors,dim=-1) > min_contribution
        waves_mask = (torch.linalg.norm(self.wave_colors,dim=-1) * \
                torch.linalg.norm(self.wave_coefficients,dim=-1)) > min_contribution
        
        if(len(gaussians_mask.shape)>0):
            to_remove = gaussians_mask.shape[0]-gaussians_mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                updated_params = self.prune_tensors_from_optimizer(gaussians_mask, "gaussian")
                self.gaussian_colors = updated_params['gaussian_colors']
                self.gaussian_means = updated_params['gaussian_means']
                self.gaussian_mats = updated_params['gaussian_mats']
        if(len(waves_mask.shape)>0):
            to_remove = waves_mask.shape[0]-waves_mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                updated_params = self.prune_tensors_from_optimizer(waves_mask, "wave")
                self.wave_colors = updated_params['wave_colors']
                self.wave_coefficients = updated_params['wave_coefficients']
                self.wave_frequencies = updated_params['wave_frequencies']
                self.wave_support_mats = updated_params['wave_support_mats']
                self.wave_support_means = updated_params['wave_support_means']

    def add_primitives(self, x, y, n_freqs=256, 
                       freq_decay = 1.0, min_influence=1./255.,
                       num_waves=1, num_gaussians=0):
        #plt.scatter(x.cpu().numpy()[:,1], x.cpu().numpy()[:,0],c=y.cpu().numpy())
        #plt.show()

        if(num_waves > 0):
            ls_model = LombScargle2D(x, y, n_terms=1, device=self.device)
            # Randomize set of wavelengths and freqs to fit in correct range
            freqs = 0.1+torch.sort(freq_decay*64*torch.rand([n_freqs], device=self.device, dtype=torch.float32)).values
            
            # Fit model and find peaks
            ls_model.fit(freqs)            
            self.data_periodogram = ls_model.get_power()
            n_extracted_waves = ls_model.find_peaks(top_n=num_waves, 
                                                    min_influence=min_influence)
            self.ls_plot = ls_model.plot_power(return_img=True)
            self.ls_power = ls_model.power
            num_gaussians += num_waves - n_extracted_waves
            if(n_extracted_waves > 0):
                all_colors = ls_model.get_PCA_color()[None,:].repeat(n_extracted_waves, 1) / n_extracted_waves

                all_frequencies = ls_model.get_peak_freqs()
                coeffs = ls_model.get_peak_coeffs()

                means, vars = ls_model.get_peak_placement(torch.arange(0,n_extracted_waves,1,dtype=torch.long,device=self.device))
                #means = means*0 + 0.5  
                mats = torch.eye(2, device=self.device, dtype=torch.float32)[None,...].repeat(n_extracted_waves, 1, 1)
                mats[:,0,0] = 1 / vars[:,0]
                mats[:,1,1] = 1 / vars[:,1]

                tensor_dict = {
                    "wave_support_means": means, 
                    "wave_support_mats": mats,
                    "wave_frequencies": all_frequencies,
                    "wave_coefficients": coeffs,
                    "wave_colors": all_colors
                }
                updated_params = self.cat_tensors_to_optimizer(tensor_dict)

                self.wave_support_means = updated_params['wave_support_means']
                self.wave_support_mats = updated_params['wave_support_mats']
                self.wave_frequencies = updated_params['wave_frequencies']
                self.wave_coefficients = updated_params['wave_coefficients']
                self.wave_colors = updated_params['wave_colors']

        else:
            n_extracted_waves = 0

        if(num_gaussians > 0):
            new_colors = 0.2*(torch.randn([num_gaussians, self.n_channels], dtype=torch.float32, device=self.device))
            new_means = torch.rand([num_gaussians, 2], dtype=torch.float32, device=self.device)
            new_mats = torch.eye(2, device=self.device, dtype=torch.float32)[None,...].repeat(num_gaussians, 1, 1) * (num_gaussians**2)
            new_mats += torch.randn_like(new_mats)*0.1

            tensor_dict = {
                "gaussian_colors": new_colors, 
                "gaussian_means": new_means,
                "gaussian_mats": new_mats,
            }

            updated_params = self.cat_tensors_to_optimizer(tensor_dict)

            self.gaussian_colors = updated_params['gaussian_colors']
            self.gaussian_means = updated_params['gaussian_means']
            self.gaussian_mats = updated_params['gaussian_mats']

        return n_extracted_waves
        
    def vis_primitives(self, x, res=[256, 256]):
        
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

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        if(self.gaussian_mats.shape[0] > 0):
            shear_loss = 0.01*((self.gaussian_mats[:,1,0]+self.gaussian_mats[:,0,1])**2).mean()
        else:
            shear_loss = 0
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
        output = 0
        if(self.wave_coefficients.shape[0] > 0):
            rel_x = x[:,None,:] - self.wave_support_means[None,...] # [N, n_gaussians, 2]

            # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
            transformed_x = ((rel_x[...,None].mT @ self.wave_support_mats[None,...])**10).sum(dim=-1, keepdim=True)

            # [N, n_gaussians, 1, 1]
            gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
            
            # periodic gaussian parts
            # Fitted sinusoidal wave
            # [N, channels]
            
            sx = torch.sin(2*torch.pi*x[:,None,0]*self.wave_frequencies[None,...,0])
            cx = torch.cos(2*torch.pi*x[:,None,0]*self.wave_frequencies[None,...,0])
            sy = torch.sin(2*torch.pi*x[:,None,1]*self.wave_frequencies[None,...,1])
            cy = torch.cos(2*torch.pi*x[:,None,1]*self.wave_frequencies[None,...,1])
            vals = self.wave_coefficients[None,:,0,0]*cx*cy + \
                    self.wave_coefficients[None,:,0,1]*cx*sy + \
                    self.wave_coefficients[None,:,0,2]*sx*cy + \
                    self.wave_coefficients[None,:,0,3]*sx*sy + \
                    self.wave_coefficients[None,:,0,4]
            
            # [N, n_primitives, 1]
            vals = gauss_vals*vals[...,None]*self.wave_colors[None,...]
            output = output + vals.sum(dim=1)

        if(self.gaussian_mats.shape[0] > 0):
            rel_x = x[:,None,:] - self.gaussian_means[None,...] # [N, n_gaussians, 2]
            transformed_x = ((rel_x[...,None].mT @ self.gaussian_mats[None,...])**2).sum(dim=-1, keepdim=True)
            gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
            gauss_vals = gauss_vals*self.gaussian_colors[None,...]
            output = output + gauss_vals.sum(dim=1)
            
        return output 

