from typing import Optional
import torch
from torch.nn.parameter import Parameter
import time
from models.LombScargle import MyLombScargleModel
from models.GaussianLombScargle import GaussianLombScargleModel
from tqdm import tqdm
from utils.data_utils import to_img
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import psnr
import tinycudann as tcnn
torch.backends.cuda.matmul.allow_tf32 = True

class PeriodicGaussianField(torch.nn.Module):

    def __init__(self, n_waves, n_features=3, n_channels=1, device="cuda"):
        super().__init__()
        # Parameters
        self.n_waves = n_waves
        self.device = device
        self.n_features = n_features
        self.n_channels = n_channels

        self.gaussian_means = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.gaussian_mats = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.subgaussian_wavelength = Parameter(torch.empty(0, device=device, dtype=torch.float32))    
        self.subgaussian_rotation = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.subgaussian_offset = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.subgaussian_flat_top_power = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.subgaussian_width = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.features = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.decoder = tcnn.Network(self.n_features, self.n_channels,
                                network_config={
                                    "otype": "FullyFusedMLP",
                                    "activation": "ReLU",
                                    "output_activation": "None",
                                    "n_neurons": 64,
                                    "n_hidden_layers": 4,
                                })
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_means], 'lr': 0.001, "name": "gaussian_means"},
            {'params': [self.gaussian_mats], 'lr': 0.001, "name": "gaussian_mats"},
            {'params': [self.subgaussian_wavelength], 'lr': 0.00001, "name": "subgaussian_wavelength"},
            {'params': [self.subgaussian_rotation], 'lr': 0.00001, "name": "subgaussian_rotation"},
            {'params': [self.subgaussian_offset], 'lr': 0.00001, "name": "subgaussian_offset"},
            {'params': [self.subgaussian_flat_top_power], 'lr': 0.00, "name": "subgaussian_flat_top_power"},
            {'params': [self.subgaussian_width], 'lr': 0.001, "name": "subgaussian_width"},
            {'params': [self.features], 'lr': 0.0001, "name": "features"},
            {'params': self.decoder.params, 'lr': 0.0001, "name": "decoder_weights"}
        ]

        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    
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

    def prune_gaussians(self, min_contribution=1./255.):
        if(self.gaussian_means.shape[0] == 0):
            return
        mask = torch.linalg.norm(self.colors,dim=-1) > min_contribution
        to_remove = mask.shape[0]-mask.sum()
        if(to_remove>0):
            #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
            updated_params = self.prune_tensors_from_optimizer(mask)

            self.colors = updated_params['colors']
            self.gaussian_means = updated_params['gaussian_means']
            self.gaussian_mats = updated_params['gaussian_mats']
            self.subgaussian_rotation = updated_params['subgaussian_rotation']
            self.subgaussian_flat_top_power = updated_params['subgaussian_flat_top_power']
            self.subgaussian_offset = updated_params['subgaussian_offset']
            self.subgaussian_width = updated_params['subgaussian_width']
            self.subgaussian_wavelength = updated_params['subgaussian_wavelength']

    def add_next_wave(self, x, y, n_waves = 1, n_freqs=512, n_angles=180, freq_decay = 1.0, min_influence=1./255.):
        #plt.scatter(x.cpu().numpy()[:,1], x.cpu().numpy()[:,0],c=y.cpu().numpy())
        #plt.show()
        ls_model = MyLombScargleModel(x, y, n_terms=1, device=self.device)
        # Randomize set of wavelengths and freqs to fit in correct range
        wavelengths = torch.sort(freq_decay*2.2*torch.rand([n_freqs], device=self.device, dtype=torch.float32)).values
        freqs = (1./wavelengths).flip([0])
        angles = [torch.pi*torch.sort(torch.rand([n_angles], device=self.device, dtype=torch.float32)).values]

        # Fit model and find peaks
        ls_model.fit(freqs, 
                        angles=angles, 
                        linear_in_frequency=False)            
        self.data_periodogram = ls_model.get_power()
        ls_model.find_peaks(top_n=n_waves, min_influence=min_influence, order_by_wavelength=False)
        #ls_model.plot_power()
        
        n_extracted_peaks = ls_model.get_peak_freq_angles().shape[0]
        if(n_extracted_peaks == 0):
            return n_extracted_peaks
        
        all_wavelengths = 1/ls_model.get_peak_freq_angles()[:,0:1]
        all_rotations = ls_model.get_peak_freq_angles()[:,1:2]
    
        #print(ls_model.get_peak_coeffs())
        all_features = 0.001*torch.randn([all_rotations.shape[0], self.n_features], device=self.device, dtype=torch.float32)
        # print(colors_peak*255)
        all_shifts = torch.atan2(ls_model.get_peak_coeffs()[:,2:3], ls_model.get_peak_coeffs()[:,1:2]) #- all_wavelengths/4
        #s2 = torch.atan2(ls_model.get_peak_coeffs()[:,5:6], ls_model.get_peak_coeffs()[:,4:5])
        #s3 = torch.atan2(ls_model.get_peak_coeffs()[:,8:9], ls_model.get_peak_coeffs()[:,7:8])
        all_wave_powers = -torch.rand([all_wavelengths.shape[0], 1], device=self.device, dtype=torch.float32)
        all_wave_widths = torch.log(all_wavelengths.clone()/4)
        
        means = 0.5+torch.zeros([n_extracted_peaks,2], device=self.device, dtype=torch.float32)      
        mats = torch.eye(2, device=self.device, dtype=torch.float32)

        tensor_dict = {
            "gaussian_means": means, 
            "gaussian_mats": mats[None,...].repeat(n_extracted_peaks, 1, 1),
            "subgaussian_wavelength": all_wavelengths,
            "subgaussian_rotation": all_rotations,
            "subgaussian_offset": all_shifts,
            "subgaussian_flat_top_power": all_wave_powers,
            "subgaussian_width": all_wave_widths,
            "features": all_features
        }
        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        #print(updated_params)
        self.gaussian_means = updated_params['gaussian_means']
        self.gaussian_mats = updated_params['gaussian_mats']
        self.subgaussian_wavelength = updated_params['subgaussian_wavelength']
        self.subgaussian_offset = updated_params['subgaussian_offset']
        self.subgaussian_flat_top_power = updated_params['subgaussian_flat_top_power']
        self.subgaussian_width = updated_params['subgaussian_width']
        self.subgaussian_rotation = updated_params['subgaussian_rotation']
        self.features = updated_params['features']

        return n_extracted_peaks
        
    def vis_each_wave(self, x, res=[256, 256], power=10):
        
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
            nwaves = max(self.gaussian_means.shape[0], self.n_waves)
            n_cols = min(nwaves, im_per_row)
            n_rows = 1+(nwaves//im_per_row)
            im = torch.zeros([n_rows*res[0], n_cols*res[1], 1])
            
            row_spot = 0
            col_spot = 0
            
            for i in range(vals.shape[1]):

                im[row_spot*res[0]:(row_spot+1)*res[0],
                   col_spot*res[1]:(col_spot+1)*res[1]] = (vals[:,i].detach().cpu().reshape(res+[1]) + 1)/2

                if((i+1) % im_per_row == 0):
                    row_spot += 1
                    col_spot = 0
                else:
                    col_spot += 1
        return to_img(im)

    def train_model(self, x, y, im_shape):
                
        print(f"Initializing Lomb-Scargle model on training data...")

        iters_per_wave = 2000
        waves_per_ls = 1
        total_iters = int(iters_per_wave*self.n_waves/waves_per_ls)
        
        num_params = []

        pre_fitting_imgs = []
        wave_imgs = []
        max_tries = 5
        tries = max_tries
        self.offset = y.mean()
        self.n_channels = y.shape[-1]
        max_ls_points = 2**17
        pct_of_data = max_ls_points / x.shape[0]
        
        t = tqdm(range(total_iters))
        for i in t:
            mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

            # image logging
            if i % 200 == 0 and i > 0:
                with torch.no_grad():
                    res = [200, 200]
                    xmin = x.min(dim=0).values
                    xmax = x.max(dim=0).values
                    g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
                    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                    img = self(g).reshape(res+[self.n_channels])
                    img = to_img(img)
                    pre_fitting_imgs.append(img)
                    #wave_img = self.vis_each_wave(x)
                    #wave_imgs.append(wave_img)

            # adding waves
            if i % iters_per_wave == 0:
                with torch.no_grad():    
                    residuals = y[mask]
                    if(i>0):
                        residuals -= self(x[mask])
                    #else:
                    #    residuals -= self.offset     
                    num_params.append(self.gaussian_means.shape[0]*4 + 1)
                    self.prune_gaussians(1./500.)
                    n_extracted_peaks = self.add_next_wave(x[mask],
                                        residuals,
                                        n_waves = waves_per_ls,
                                        n_freqs = 256, 
                                        n_angles = 128, 
                                        freq_decay=0.98**((i//iters_per_wave)*waves_per_ls), 
                                        min_influence=1./500.)
                    if(n_extracted_peaks == 0 and tries == 0):
                        print(f" Stopping wave detection early, no peaks found in {max_tries} iterations.")
                        break
                    elif(n_extracted_peaks == 0):
                        tries -= 1
                    else:
                        tries = max_tries
            
            # actual training step
            self.optimizer.zero_grad()
            loss, model_out = self.loss(x[mask], y[mask])
            if (loss.isnan()):
                print()
                print(f"Detected loss was NaN.")
                print(self.gaussian_means)
                print(self.gaussian_mats)
                print(self.subgaussian_flat_top_power)
                print(self.subgaussian_offset)
                print(self.subgaussian_rotation)
                print(self.subgaussian_wavelength)
                print(self.subgaussian_width)
                print(self.features)
                quit()
                
            loss.backward()
            #print(f"{self.subgaussian_width} {self.subgaussian_width.grad}")
            self.optimizer.step()
            with torch.no_grad():
                self.subgaussian_flat_top_power.clamp_(-1, 3)
            # logging
            with torch.no_grad():             
                t.set_description(f"[{i+1}/{total_iters}] loss: {loss.item():0.04f}")
        #print(p.key_averages().table(
        #    sort_by="self_cuda_time_total", row_limit=-1))
        
        imageio.imwrite("output/gaussianfield_training.mp4", pre_fitting_imgs)
        #imageio.imwrite("output/gaussianfield_training.mp4", wave_imgs)
        self.prune_gaussians(1./500.)
        print(f"Number of extracted waves: {self.gaussian_means.shape[0]}")
        output = self(x)
        p = psnr(output,y).item()
        print(f"Final PSNR: {p:0.02f}")
        
        num_params.append(self.gaussian_means.shape[0]*4+1)

        '''
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(num_params, psnrs, color="blue")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_xlabel("Num params")        
        ax1.set_title("Reconstruction vs # params")
        plt.savefig("output/training_details.png")
        plt.clf()
        '''
        err = torch.clamp(((y-output)**2), 0., 1.)
        print(err.min())
        print(err.max())
        plt.scatter(x.detach().cpu().numpy()[:,1],
                    -x.detach().cpu().numpy()[:,0],
                    c=err.detach().cpu().numpy().reshape(-1, self.n_channels))
        plt.savefig("./output/supportedperiodicprims_output.png")


        #print(self.means)
        #print(self.gaussian_mats)

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
        l1 = torch.nn.functional.mse_loss(model_out,y)
        shear_loss = 0.01*((self.gaussian_mats[:,1,0]+self.gaussian_mats[:,0,1])**2).mean()
        final_loss = l1 + shear_loss#+ decay_loss
        return final_loss, model_out

    def forward(self, x: torch.Tensor, sum_together:Optional[bool]=True):
        
        # x is [N, 2]
        # gaussian coeffs
        rel_x = x[:,None,:] - self.gaussian_means[None,...] # [N, n_gaussians, 2]
        RS = self.create_RS() # [n_gaussians, 2, 2]
        cov = RS #@ RS.mT
        #cov = inv2x2(cov)
        # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
        #transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
        transformed_x = ((rel_x[...,None].mT @ cov[None,...])**10).sum(dim=-1, keepdim=True)
        # [N, n_gaussians, 1, 1]
        gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
        
        # periodic gaussian parts
        rotated_x = x[:,None,0:1]*torch.cos(self.subgaussian_rotation[None,...]) + \
                    x[:,None,1:2]*torch.sin(self.subgaussian_rotation[None,...])
        
        # Fitted sinusoidal wave
        vals = torch.sin(2*torch.pi*rotated_x/self.subgaussian_wavelength[None,...] + self.subgaussian_offset[None,...])
        # Rescale 0-1
        vals = 1 - (vals + 1)/2
        # Adjust wave width
        vals = vals / torch.exp(self.subgaussian_width[None,...])
        # [N, n_waves, 1]
        #vals = (rotated_x + self.subgaussian_wavelength[None,...].detach()/4 + \
        #        ((self.subgaussian_wavelength[None,...].detach()*self.subgaussian_offset[None,...])/(2*torch.pi))) \
        #    % self.subgaussian_wavelength[None,...]
        #vals = vals - self.subgaussian_wavelength[None,...].detach()/2
        #vals = vals ** 6
        #vals = vals / (torch.exp(self.subgaussian_width[None,...])**2)
        vals = vals ** (1+torch.exp(self.subgaussian_flat_top_power[None,...]))
        vals = torch.exp(-vals)
        
        if(sum_together):
            vals = gauss_vals*vals*self.features[None,...]
            vals = vals.sum(dim=1)
            vals = self.decoder(vals).float()
        else:
            vals = gauss_vals*vals
        return vals 

