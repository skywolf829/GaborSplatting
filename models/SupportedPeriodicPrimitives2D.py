import torch
from torch.nn.parameter import Parameter
from utils.data_utils import rgb2hsv_torch, hsv2rgb_torch
import time
from models.LombScargle import MyLombScargleModel
from tqdm import tqdm
from utils.data_utils import to_img, psnr
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from models.GaussianSplatting2D import inv2x2
torch.backends.cuda.matmul.allow_tf32 = True

class SupportedPeriodicPrimitives2D(torch.nn.Module):

    def __init__(self, n_waves, n_terms = 1, device="cuda"):
        super().__init__()
        # Parameters
        self.n_waves = n_waves
        self.n_terms = n_terms
        self.device = device

        self.freqs = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.rotations = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.shifts = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.wave_powers = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.colors_peaks = Parameter(torch.empty(0, device=device, dtype=torch.float32))

        self.means = Parameter(torch.empty(0, device=device, dtype=torch.float32))    
        self.gaussian_mats = Parameter(torch.empty(0, device=device, dtype=torch.float32))    

        self.register_buffer("offset", torch.empty(0, device=device, dtype=torch.float32))
        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.freqs], 'lr': 0.001, "name": "freqs"},
            {'params': [self.rotations], 'lr': 0.001, "name": "rotations"},
            {'params': [self.shifts], 'lr': 0.001, "name": "shifts"},
            {'params': [self.wave_powers], 'lr': 0.001, "name": "wave_powers"},
            
            {'params': [self.colors_peaks], 'lr': 0.001, "name": "colors_peaks"},
            
            {'params': [self.means], 'lr': 0.001, "name": "means"},            
            {'params': [self.gaussian_mats], 'lr': 0.001, "name": "gaussian_mats"}
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
        #plt.scatter(x.cpu().numpy()[:,1], x.cpu().numpy()[:,0],c=y.cpu().numpy())
        #plt.show()
        ls_model = MyLombScargleModel(x, y, n_terms=self.n_terms, device=self.device)
        # Randomize set of wavelengths and freqs to fit in correct range
        wavelengths = torch.sort(freq_decay*2*torch.rand([n_freqs], device=self.device, dtype=torch.float32)).values
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
        all_freqs = ls_model.get_peak_freq_angles()[:,0:1]
        all_rotations = ls_model.get_peak_freq_angles()[:,1:2]
    
        #print(ls_model.get_peak_coeffs())
        r_peak = torch.linalg.norm(ls_model.get_peak_coeffs()[:,1:3:3], dim=-1, keepdim=True)+ls_model.get_peak_coeffs()[:,0:1:3]
        g_peak = torch.linalg.norm(ls_model.get_peak_coeffs()[:,4:6], dim=-1, keepdim=True)+ls_model.get_peak_coeffs()[:,3:4]
        b_peak = torch.linalg.norm(ls_model.get_peak_coeffs()[:,6:9], dim=-1, keepdim=True)+ls_model.get_peak_coeffs()[:,6:7]
        colors_peak = torch.cat([r_peak,g_peak,b_peak], dim=-1) / n_waves
        # print(colors_peak*255)
        all_shifts = torch.atan2(ls_model.get_peak_coeffs()[:,2:3], ls_model.get_peak_coeffs()[:,1:2])
        #s2 = torch.atan2(ls_model.get_peak_coeffs()[:,5:6], ls_model.get_peak_coeffs()[:,4:5])
        #s3 = torch.atan2(ls_model.get_peak_coeffs()[:,8:9], ls_model.get_peak_coeffs()[:,7:8])
        all_wave_powers = torch.zeros([all_freqs.shape[0], 1], device=self.device, dtype=torch.float32)

        
        means = 0.5+torch.zeros([2], device=self.device, dtype=torch.float32)      
        mats = torch.eye(2, device=self.device, dtype=torch.float32)

        # top of table  
        #mats = torch.tensor([[6.2389,  2.5744],
        #                    [-2.5636,  1.]], device=self.device, dtype=torch.float32)
        #means = torch.tensor([0.3, 0.3], device=self.device, dtype=torch.float32)

        tensor_dict = {
            "freqs": all_freqs, 
            "rotations": all_rotations,
            "colors_peaks": colors_peak,
            "shifts": all_shifts,
            "wave_powers": all_wave_powers,
            "means": means[None,...].repeat(all_freqs.shape[0], 1),
            "gaussian_mats": mats[None,...].repeat(all_freqs.shape[0],1,1)
        }
        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        #print(updated_params)
        self.freqs = updated_params['freqs']
        self.rotations = updated_params['rotations']
        self.shifts = updated_params['shifts']
        self.wave_powers = updated_params['wave_powers']

        self.colors_peaks = updated_params['colors_peaks']

        self.means = updated_params['means']
        self.gaussian_mats = updated_params['gaussian_mats']
        return n_extracted_peaks

    def add_next_wave_nols(self, x, y, n_waves = 1, n_freqs=512, n_angles=180, freq_decay = 1.0, min_influence=1./255.):
        #plt.scatter(x.cpu().numpy()[:,1], x.cpu().numpy()[:,0],c=y.cpu().numpy())
        #plt.show()

        all_freqs = torch.rand([n_waves, 1], device=self.device, dtype=torch.float32)
        all_rotations = torch.rand([n_waves, 1], device=self.device, dtype=torch.float32)
    
        #print(ls_model.get_peak_coeffs())
        colors_peak = torch.rand([n_waves, self.n_channels], device=self.device, dtype=torch.float32) / n_waves
        all_shifts = 2*torch.rand([n_waves, 1], device=self.device, dtype=torch.float32)-1
        all_wave_powers = torch.zeros([all_freqs.shape[0], 1], device=self.device, dtype=torch.float32)

        means = 0.5+torch.zeros([2], device=self.device, dtype=torch.float32)      
        mats = torch.eye(2, device=self.device, dtype=torch.float32)

        # top of table  
        #mats = torch.tensor([[6.2389,  2.5744],
        #                    [-2.5636,  1.]], device=self.device, dtype=torch.float32)
        #means = torch.tensor([0.3, 0.3], device=self.device, dtype=torch.float32)

        tensor_dict = {
            "freqs": all_freqs, 
            "rotations": all_rotations,
            "colors_peaks": colors_peak,
            "shifts": all_shifts,
            "wave_powers": all_wave_powers,
            "means": means[None,...].repeat(all_freqs.shape[0], 1),
            "gaussian_mats": mats[None,...].repeat(all_freqs.shape[0],1,1)
        }
        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        #print(updated_params)
        self.freqs = updated_params['freqs']
        self.rotations = updated_params['rotations']
        self.shifts = updated_params['shifts']
        self.wave_powers = updated_params['wave_powers']

        self.colors_peaks = updated_params['colors_peaks']

        self.means = updated_params['means']
        self.gaussian_mats = updated_params['gaussian_mats']
        return n_waves

    def prune_waves(self, min_contribution=1./255.):
        mask = (self.colors > min_contribution) + (self.wave_offsets > min_contribution)
        mask = mask.squeeze()
        if(len(mask.shape)>0):
            to_remove = mask.shape[0]-mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                optimizable_tensors = self.prune_tensors_from_optimizer(mask)

                self.colors = optimizable_tensors["colors"]
                self.freqs = optimizable_tensors["freqs"]
                self.rotations = optimizable_tensors["rotations"]                
                self.means = optimizable_tensors['means']
                self.wave_powers = optimizable_tensors['wave_powers']
                self.shifts = optimizable_tensors['shifts']
                self.gaussian_mats = optimizable_tensors['gaussian_mats']
                self.wave_offsets = optimizable_tensors['wave_offsets']

    def vis_each_wave(self, x, res=[256, 256], power=10):
        
        with torch.no_grad():
            xmin = x.min(dim=0).values
            xmax = x.max(dim=0).values
            g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            x = g
            # x is [N, 2]
            # gaussian coeffs
            rel_x = x[:,None,:] - self.means[None,...] # [N, n_gaussians, 2]
            RS = self.create_RS() # [n_gaussians, 2, 2]
            cov = RS #@ RS.mT
            #cov = inv2x2(cov)
            # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
            #transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
            transformed_x = ((rel_x[...,None].mT @ cov[None,...])**power).sum(dim=-1, keepdim=True)
            # [N, n_gaussians, 1, 1]
            gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
            
            # sine parts
            rotated_x = x[:,None,0:1]*torch.cos(self.rotations)[None,:] + \
                        x[:,None,1:2]*torch.sin(self.rotations)[None,:]
            # [N, n_waves, 1]
            
            vals = (1+torch.sin(2*torch.pi*rotated_x*self.freqs[None,...] + self.shifts))/2
            #vals = torch.sign(vals)*(torch.abs(vals)**torch.exp(self.wave_powers))
            vals = self.colors_peaks[None,...]*vals #+ (1-vals)*self.colors_valleys[None,...]
            vals = gauss_vals*vals

            im_per_row = 4
            nwaves = max(self.means.shape[0], self.n_waves)
            n_cols = min(nwaves, im_per_row)
            n_rows = 1+(nwaves//im_per_row)
            im = torch.zeros([n_rows*res[0], n_cols*res[1], self.n_channels])
            
            row_spot = 0
            col_spot = 0
            
            for i in range(vals.shape[1]):

                im[row_spot*res[0]:(row_spot+1)*res[0],
                   col_spot*res[1]:(col_spot+1)*res[1]] = (vals[:,i].detach().cpu().reshape(res+[self.n_channels]) + 1)/2

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
        psnrs = []

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
            if i % 500 == 0 and i > 0:
                with torch.no_grad():
                    res = [200, 200]
                    xmin = x.min(dim=0).values
                    xmax = x.max(dim=0).values
                    g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
                    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                    img = self(g).reshape(res+[self.n_channels])
                    img = to_img(img)
                    pre_fitting_imgs.append(img)
                    wave_img = self.vis_each_wave(x)
                    wave_imgs.append(wave_img)

            # adding waves
            if i % iters_per_wave == 0:
                with torch.no_grad():    
                    residuals = y[mask]
                    if(i>0):
                        residuals -= self(x[mask])
                    #else:
                    #    residuals -= self.offset     
                    num_params.append(self.means.shape[0]*4 + 1)
                    #self.prune_waves(1./500.)
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
                print(f"Detected loss was NaN. Parameters were:")
                print(f"Gaussian means: {self.means}")
                print(f"Gaussian matrices: {self.gaussian_mats}")
                print(f"Frequencies: {self.freqs}")
                print(f"Angles: {self.rotations}")
                print(f"Shifts: {self.shifts}")
                print(f"Wave powers: {self.wave_powers}")
                print(f"Offsets: {self.offset}")
                quit()
                
            loss.backward()
            self.optimizer.step()

            # logging
            with torch.no_grad():             
                t.set_description(f"[{i+1}/{total_iters}] loss: {loss.item():0.04f}")
        #print(p.key_averages().table(
        #    sort_by="self_cuda_time_total", row_limit=-1))
        
        imageio.imwrite("output/supported_waves_training_err.mp4", pre_fitting_imgs)
        imageio.imwrite("output/supported_waves_training.mp4", wave_imgs)
        #self.prune_waves(1./500.)
        print(f"Number of extracted waves: {self.freqs.shape[0]}")
        output = self(x)
        p = psnr(output,y).item()
        print(f"Final PSNR: {p:0.02f}")
        
        num_params.append(self.means.shape[0]*4+1)

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

    def create_rotation_matrices(self):
        return torch.stack([torch.cos(self.gaussian_rotations), -torch.sin(self.gaussian_rotations),
                             torch.sin(self.gaussian_rotations), torch.cos(self.gaussian_rotations)], dim=-1).reshape(-1, 2, 2)
    
    def create_RS(self):
        #return torch.stack([torch.exp(self.scales[:,0:1])*torch.cos(self.gaussian_rotations), 
        #                              torch.exp(self.scales[:,1:2])*-torch.sin(self.gaussian_rotations),
        #                    torch.exp(self.scales[:,0:1])*torch.sin(self.gaussian_rotations), 
        #                              torch.exp(self.scales[:,1:2])*torch.cos(self.gaussian_rotations)], 
        #                    dim=-1).reshape(-1, 2, 2)
        return self.gaussian_mats

    def loss(self, x, y, detach_waves=False):
        # x is our output, y is the ground truth
        model_out = self(x, detach_waves=detach_waves)
        l1 = torch.nn.functional.mse_loss(model_out,y)
        shear_loss = 0.01*((self.gaussian_mats[:,1,0]+self.gaussian_mats[:,0,1])**2).mean()
        final_loss = l1 + shear_loss#+ decay_loss
        return final_loss, model_out

    def forward(self, x, detach_waves=False, power=10):
        # x is [N, 2]
        # gaussian coeffs
        rel_x = x[:,None,:] - self.means[None,...] # [N, n_gaussians, 2]
        RS = self.create_RS() # [n_gaussians, 2, 2]
        cov = RS #@ RS.mT
        #cov = inv2x2(cov)
        # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
        #transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
        transformed_x = ((rel_x[...,None].mT @ cov[None,...])**power).sum(dim=-1, keepdim=True)
        # [N, n_gaussians, 1, 1]
        gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
        
        # sine parts
        rotated_x = x[:,None,0:1]*torch.cos(self.rotations)[None,:] + \
                    x[:,None,1:2]*torch.sin(self.rotations)[None,:]
        # [N, n_waves, 1]
        vals = (1+torch.sin(2*torch.pi*rotated_x*self.freqs[None,...] + self.shifts))/2
        c = vals*self.colors_peaks[None,...]# + (1-vals)*self.colors_valleys[None,...]
        
        #vals = torch.sign(vals)*(torch.abs(vals)**torch.exp(self.wave_powers))
        vals = (gauss_vals*c).sum(dim=1)
        return vals #+ self.offset # [N, channels]

