import torch
from torch.nn.parameter import Parameter
import time
from models.LombScargle import MyLombScargleModel
from tqdm import tqdm
from utils.data_utils import to_img, psnr
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
torch.backends.cuda.matmul.allow_tf32 = True

class PeriodicPrimitives2D(torch.nn.Module):

    def __init__(self, n_waves, n_channels = 1, n_terms = 1, device="cuda"):
        super().__init__()
        # Parameters
        self.n_waves = n_waves
        self.device = device
        self.n_channels = n_channels

        self.freqs = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.rotations = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.coeffs = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.offsets = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.n_terms = n_terms
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.freqs], 'lr': 0.000001, "name": "freqs"},
            {'params': [self.rotations], 'lr': 0.000001, "name": "rotations"},
            {'params': [self.coeffs], 'lr': 0.000001, "name": "coeffs"},
            {'params': [self.offsets], 'lr': 0.00001, "name": "offsets"}
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

    def add_next_wave(self, x, y, n_waves = 1, n_freqs=512, n_angles=180, freq_decay = 1.0, min_influence=1./255.):

        max_ls_points = 2**17
        pct_of_data = max_ls_points / x.shape[0]
        mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data
        ls_model = MyLombScargleModel(x[mask], y[mask], device=self.device)
        # Randomize set of wavelengths and freqs to fit in correct range
        wavelengths = torch.sort(freq_decay*3*torch.rand([n_freqs], device=self.device, dtype=torch.float32)).values
        freqs = (1./wavelengths).flip([0])
        angles = [torch.pi*torch.sort(torch.rand([n_angles], device=self.device, dtype=torch.float32)).values]

        # Fit model and find peaks
        ls_model.fit(freqs, 
                        angles=angles, 
                        linear_in_frequency=False)            
        self.data_periodogram = ls_model.get_power()
        ls_model.find_peaks(top_n=n_waves, min_influence=min_influence)
        #ls_model.plot_power()

        n_extracted_peaks = ls_model.get_peak_freq_angles().shape[0]
        if(n_extracted_peaks == 0):
            return n_extracted_peaks
        all_freqs = ls_model.get_peak_freq_angles()[:,0:1]
        all_rotations = ls_model.get_peak_freq_angles()[:,1:2]
        all_coeffs = ls_model.get_peak_coeffs()[:,1:]
        all_offsets =  ls_model.get_peak_coeffs()[:,0:1]

        tensor_dict = {
            "freqs": all_freqs, 
            "rotations": all_rotations,
            "coeffs": all_coeffs,
            "offsets": all_offsets
        }

        updated_params = self.cat_tensors_to_optimizer(tensor_dict)
        self.freqs = updated_params['freqs']
        self.rotations = updated_params['rotations']
        self.coeffs = updated_params['coeffs']
        self.offsets = updated_params['offsets']
        return n_extracted_peaks

    def prune_waves(self, min_contribution=1./255.):
        mask = torch.linalg.norm(self.coeffs, dim=-1) > min_contribution
        if(len(mask.shape)>0):
            to_remove = mask.shape[0]-mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                optimizable_tensors = self.prune_tensors_from_optimizer(mask)

                self.coeffs = optimizable_tensors["coeffs"]
                self.freqs = optimizable_tensors["freqs"]
                self.rotations = optimizable_tensors["rotations"]

    def train_model(self, x, y, im_shape):
                
        print(f"Initializing Lomb-Scargle model on training data...")

        iters_per_wave = 500
        n_extracted_peaks = 1
        total_iters = int(iters_per_wave*self.n_waves/n_extracted_peaks)
        
        p = 0

        num_params = []
        psnrs = []

        pre_fitting_imgs = []
        max_tries = 5
        tries = max_tries
        max_ls_points = 2**17
        pct_of_data = max_ls_points / x.shape[0]
        
        t = tqdm(range(total_iters))
        for i in t:
            mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

            # image logging
            if i % 100 == 0 and i > 0:
                with torch.no_grad():
                    res = [200, 200]
                    xmin = x.min(dim=0).values
                    xmax = x.max(dim=0).values
                    g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
                    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                    img = self(g).reshape(res)
                    img = to_img(img)
                    pre_fitting_imgs.append(img)

            # adding waves
            if i % iters_per_wave == 0:
                with torch.no_grad():    
                    residuals = y[mask]
                    if(i>0):
                        residuals -= self(x[mask])                     
                    psnrs.append(p)
                    num_params.append(self.coeffs.shape[0]*4 + 1)
                    self.prune_waves(1./500.)
                    n_extracted_peaks = self.add_next_wave(x[mask],
                                        residuals,
                                        n_waves = 1,
                                        n_freqs = 128, 
                                        n_angles = 128, 
                                        freq_decay=0.99**(((i*n_extracted_peaks)//iters_per_wave)-1), 
                                        min_influence=1./200.)
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
            loss.backward()
            self.optimizer.step()

            # logging
            with torch.no_grad():             
                t.set_description(f"[{i+1}/{total_iters}] loss: {loss.item():0.04f}, PSNR: {p:0.02f}")

        imageio.imwrite("output/prefitting.gif", pre_fitting_imgs)
        self.prune_waves(1./500.)
        print(f"Number of extracted waves: {self.freqs.shape[0]}")
        p = psnr(self(x),y).item()
        print(f"Final PSNR: {p:0.02f}")
        
        psnrs.append(p)
        num_params.append(self.coeffs.shape[0]*4+1)

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(num_params, psnrs, color="blue")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_xlabel("Num params")        
        ax1.set_title("Reconstruction vs # params")
        plt.savefig("output/training_details.png")
        
    def increase_frequencies(self, x, y):

        imgs = []
        with torch.no_grad():
            for i in np.linspace(1.0, 4.0, 100):
                self.freqs *= i

                res = [256, 256]
                xmin = x.min(dim=0).values
                xmax = x.max(dim=0).values
                g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
                g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                img = self(g).reshape(res)
                img = to_img(img)
                imgs.append(img)

                self.freqs /= i
        imageio.imwrite("output/freq_changing.gif", imgs)


    def create_rotation_matrices(self):
        return torch.stack([torch.cos(self.rotations), -torch.sin(self.rotations),
                             torch.sin(self.rotations), torch.cos(self.rotations)], dim=-1).reshape(-1, 2, 2)

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        l1 = torch.nn.functional.l1_loss(model_out,y)
        decay_loss = 0.1*torch.linalg.norm(self.coeffs, dim=-1).mean()
        final_loss = l1 #+ decay_loss
        return final_loss, model_out

    def forward(self, x):
        # x is [N, 2]
        rotated_x = x[:,None,0:1]*torch.cos(self.rotations)[None,:] + \
                    x[:,None,1:2]*torch.sin(self.rotations)[None,:]
        
        # [N, n_waves, 1]
        vals =  self.coeffs[None,:,0:1]*torch.sin(2*torch.pi*rotated_x*self.freqs[None,:]) + \
                self.coeffs[None,:,1:2]*torch.cos(2*torch.pi*rotated_x*self.freqs[None,:]) + \
                self.offsets[None,:,0:1]
        vals = vals.sum(dim=1)
        return vals# [N, channels]

