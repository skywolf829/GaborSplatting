import torch
import matplotlib.pyplot as plt
import os
from typing import List
import numpy as np
from tqdm import tqdm
from time import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cuda.matmul.allow_tf32 = True

class LombScargle2D():
    def __init__(self, x: torch.Tensor, y:torch.Tensor, n_terms = 1, device="cuda"):
        if(device == "cuda"):
            assert torch.cuda.is_available(), f"Set device is cuda, but torch.cuda.is_available() = False"
        assert torch.is_tensor(x), f"x is not a tensor"
        assert torch.is_tensor(y), f"y is not a tensor"    
        assert x.ndim == 2, f"x requires 2 dimensions but found |{x.shape}|={x.ndim}"
        assert y.ndim == 2, f"y requires 2 dimensions but found |{y.shape}|={y.ndim}"
        assert y.shape[0] == x.shape[0], f"Dimension 0 of x and y should be the same. Found x:{x.shape[0]}, y:{y.shape[0]}"
        assert x.shape[1] == 1 or x.shape[1] == 2 or x.shape[1] == 3, f"Only supports 1D, 2D, and 3D, not {x.shape[1]}-D"

        self.device = device
        self.x : torch.Tensor = x.to(device)
        self.y : torch.Tensor = y.to(device)

        self.n_items : int = x.shape[0]
        self.n_dims : int = x.shape[1]
        self.n_bands : int = y.shape[1]
        self.unique_bands = torch.arange(self.n_bands, device=device)

        self.modeled_frequencies : torch.Tensor = None

        self.wave_coefficients : torch.Tensor = None
        self.power : torch.Tensor = None

        self.peak_idx : torch.Tensor = None

        self.linear_in_frequency : bool = False
        self.perfect_fit : bool = False
        
        self.nterms = n_terms
        self.nterms_band = 1
        self.reg_band = 1e-6

    def generate_waves(self, x, y):
        x = x[None,...]
        y = y[:,None,...]
        waves = torch.empty([y.shape[0], x.shape[1], x.shape[2], 9], device=x.device, dtype=torch.float32)
        waves[...,0] = torch.sin(x)*torch.sin(y)
        waves[...,1] = torch.sin(x)*torch.cos(y)
        waves[...,2] = torch.sin(x)
        waves[...,3] = torch.cos(x)*torch.sin(y)
        waves[...,4] = torch.cos(x)*torch.cos(y)
        waves[...,5] = torch.cos(x)
        waves[...,6] = torch.sin(y)
        waves[...,7] = torch.cos(y)
        waves[...,8] = 1

        return waves

    def fit(self, frequencies:torch.Tensor):
        """
        Fits the model to each frequency in frequencies.
        Also fits angles, which are given as a list of angles in theta, phi order.
        """
        
        assert torch.is_tensor(frequencies), f"frequencies is not a tensor"
        assert frequencies.ndim == 1, f"frequencies should only have 1 dim, found {frequencies.shape}"

        self.peak_idx = None
        
        frequencies = frequencies.to(self.device)

        self.modeled_frequencies = frequencies
        self.wave_coefficients = torch.empty([frequencies.shape[0], 
                                              frequencies.shape[0], 
                                              1, 9],
                                              device=self.device, dtype=torch.float32)
        self.power = torch.empty([frequencies.shape[0], frequencies.shape[0]], 
                                 device = self.device, dtype=torch.float32)

        pca_result = torch.pca_lowrank(self.y, center=True)
        proj_y = self.y @ pca_result[2][:,:1]
        #plt.scatter(self.x[:,1].cpu().numpy(), -self.x[:,0].cpu().numpy(), 
        # c=proj_y.cpu().numpy(), cmap="gray")
        #plt.show()
        max_col = torch.argmax(proj_y)
        self.PCA_color = self.y[max_col]
        self.avg_color = self.y.mean(dim=0)
        print(self.PCA_color*255)
        # Construct weighted y matrix by subtracting means for each band
        yw = proj_y

        # Calculate chi-squared
        chi2 = yw.t() @ yw  # reference chi2 for later comparison

        max_system_size = 2**19
        rows_per_batch = max(1, min(frequencies.shape[0], 
                        int(frequencies.shape[0]*max_system_size/(frequencies.shape[0]*self.x.shape[0]))))
        start_time = time()        
        with torch.no_grad():
            row = 0
            while row < frequencies.shape[0]:         
                end_row = min(row+rows_per_batch, frequencies.shape[0])   
                # Construct X - design matrix of the stationary sinusoid model
                # Most time is spent making this.
                x_modulated = 2*torch.pi*self.x[None,:,0] * self.modeled_frequencies[:,None]
                y_modulated = 2*torch.pi*self.x[None,:,1] * self.modeled_frequencies[row:end_row,None]
                
                X = self.generate_waves(x_modulated, y_modulated)
                XTX = X.mT @ X
                XTy = X.mT @ yw[None,...]
                # Matrix Algebra to calculate the Lomb-Scargle power at each omega step
                theta_MLE = torch.linalg.solve(XTX, XTy)
                del X
                del XTX
                # update model
                self.wave_coefficients[row:end_row] = theta_MLE.mT

                # chi-squared for power
                p = (XTy.mT @ theta_MLE)/chi2
                del XTy
                p = torch.diagonal(p, dim1=-2, dim2=-1).mean(dim=-1)
                self.power[row:end_row] = p
                row = end_row
        end_time = time()
        #print(f"Fitting took {end_time-start_time:0.02f} sec")

    def to_two_wave_form(self, coeffs):
        # coeffs [N,channels,9]
        # represent adsin(x)sin(y)+aesin(x)cos(y)+afcos(x)sin(y)+bdcos(x)cos(y)+...+cf
        # finds abcdef separated
        abcdef = torch.empty([coeffs.shape[0], coeffs.shape[1], 6], 
                            device=self.device, dtype=torch.float32)

        abcdef[:,:,0] = coeffs[:,:,0] / coeffs[:,:,3] 
        abcdef[:,:,1] = 1
        abcdef[:,:,2] = coeffs[:,:,6] / coeffs[:,:,3] 
        abcdef[:,:,3] = coeffs[:,:,3]
        abcdef[:,:,4] = coeffs[:,:,4]
        abcdef[:,:,5] = coeffs[:,:,5]
        return abcdef

    def to_one_wave_form(self, coeffs):
        # coeffs [N,channels,9]
        # represent (asin(x)+bcos(x)+c)(dsin(y)+ecos(y)+f)
        # finds (asin(x+b)+c)(dsin(y+e)+f)
        abcdef = torch.empty([coeffs.shape[0], coeffs.shape[1], 6], 
                            device=self.device, dtype=torch.float32)
        abcdef[:,:,0] = torch.linalg.norm(coeffs[:,:,0:2], dim=-1)
        abcdef[:,:,1] = torch.atan2(coeffs[:,:,1], coeffs[:,:,0])
        abcdef[:,:,2] = coeffs[:,:,2]
        abcdef[:,:,3] = torch.linalg.norm(coeffs[:,:,3:5], dim=-1)
        abcdef[:,:,4] = torch.atan2(coeffs[:,:,4], coeffs[:,:,3])
        abcdef[:,:,5] = coeffs[:,:,5]

        return abcdef
    
    def get_power(self):
        assert self.power is not None, f"power is none, fit a model first"
        return self.power
    
    def get_wave_coefficients(self):
        assert self.wave_coefficients is not None, f"wave_coefficients is none, fit a model first"
        return self.wave_coefficients
    
    def find_peaks(self, min_influence=1./255., top_n = None):
        assert self.power is not None, f"power is none, fit a model first"
        kernel_size = 29
        padding = kernel_size//2

        if(self.n_dims == 1):
            filter = torch.nn.MaxPool1d(kernel_size, padding=0, stride=1)
        elif(self.n_dims == 2):
            filter = torch.nn.MaxPool2d(kernel_size, padding=0, stride=1)
        elif(self.n_dims == 3):
            filter = torch.nn.MaxPool3d(kernel_size, padding=0, stride=1)
        else:
            print(f"Peak finding not supported for {self.n_dims}D")
        
        # Pad with circular padding for angle, constant for freq.
        sig = torch.nn.functional.pad(self.power[None,None,...], pad=[0, 0, padding, padding], mode='circular')
        sig = torch.nn.functional.pad(sig, pad=[padding, padding, 0, 0], mode='constant', value=-float("Inf"))
        maxes = filter(sig)
        criterion = torch.linalg.norm(self.wave_coefficients.flatten(-2, -1),dim=-1) > min_influence
        peaks = (self.power == maxes[0,0]) * criterion
        self.peak_idx = torch.argwhere(peaks)
        if(self.n_dims == 1):
            powers = self.power[self.peak_idx[:,0]]
        elif(self.n_dims == 2):
            powers = self.power[self.peak_idx[:,0], self.peak_idx[:,1]]
        
        ordered_power = torch.argsort(powers, descending=True)
        powers = powers[ordered_power]
        self.peak_idx = self.peak_idx[ordered_power]

        if top_n is not None:            
            self.peak_idx = self.peak_idx[0:min(top_n, self.peak_idx.shape[0])]
        #self.peak_idx[:,0] = self.modeled_frequencies.shape[0]-1-self.peak_idx[0]
        #print(f"Extracted {self.peak_idx.shape[0]} peaks.")
        return self.peak_idx.shape[0]

    def get_PCA_color(self):
        return self.PCA_color
    
    def get_peak_freqs(self):
        assert self.peak_idx is not None, "Must compute peaks first with find_peaks()"

        if(self.n_dims == 1):
            freqs = self.modeled_frequencies[self.peak_idx[:,0]]
            return freqs
        elif(self.n_dims == 2):
            freq_x = self.modeled_frequencies[self.peak_idx[:,1]]
            freq_y = self.modeled_frequencies[self.peak_idx[:,0]]
            return torch.stack([freq_x, freq_y], dim=-1)

    def get_peak_coeffs(self):
        assert self.peak_idx is not None, "Must compute peaks first with find_peaks()"
        coeffs = self.wave_coefficients[self.peak_idx[:,0], self.peak_idx[:,1], :]
        return coeffs

    def get_peak_power(self):
        assert self.peak_idx is not None, "Must compute peaks first with find_peaks()"
        coeffs = self.power[self.peak_idx[:,0], self.peak_idx[:,1]]
        return coeffs

    def get_num_peaks(self):
        assert self.peak_idx is not None, f"Must find peaks first"
        return self.peak_idx.shape[0]

    def plot_power(self):
        assert self.modeled_frequencies is not None, f"Need modeled frequencies"
        assert self.power is not None, f"Requires computed powers"

        if(self.n_dims == 1):
            x_axis = self.modeled_frequencies.clone()
            values = self.power.clone()
            x_label = "Frequencies (Hz)"
            y_label = "Power"
            plt.plot(x_axis.cpu().numpy(), values.cpu().numpy())

            if(self.peak_idx is not None):
                idx_peaks = self.peak_idx.clone()
                x = x_axis[idx_peaks[:,0]]
                y = values[idx_peaks[:,0]]
                plt.scatter(x.cpu().numpy(), y.cpu().numpy())

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()

        elif(self.n_dims == 2):
            x_axis = self.modeled_frequencies.clone()
            y_axis = self.modeled_frequencies.clone()
            values = self.power.clone()
            y_label = "y Frequencies (Hz)"
            x_label = "x Frequencies (Hz)"
            
            x_min = x_axis[0].cpu().numpy()
            x_max = x_axis[-1].cpu().numpy()
            y_min = y_axis[0].cpu().numpy()
            y_max = y_axis[-1].cpu().numpy()
            x_width = (x_max-x_min)
            y_height = (y_max-y_min)

            plt.imshow(values.cpu().flip(dims=[0]).numpy(), extent=[x_min, x_max, y_min, y_height], 
               aspect=x_width/y_height)
            if(self.peak_idx is not None):
                idx_peaks = self.peak_idx.clone().cpu()
                x = x_axis[idx_peaks[:,1]]
                y = y_axis[idx_peaks[:,0]]
                plt.scatter(x.cpu().numpy(), y.cpu().numpy())

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()

    def transform_from_peaks(self, new_points, top_n_peaks = None, peaks_to_use = None):
        assert torch.is_tensor(new_points), f"new_points must be a tensor"
        assert self.peak_idx is not None, "peak_idx cannot be None"
        assert self.power is not None, "power cannot be None"
        if(peaks_to_use is not None):
            for i in range(len(peaks_to_use)):
                assert peaks_to_use[i] >= 0 and peaks_to_use[i] < self.peak_idx.shape[0]
        new_points = new_points.to(self.device)
        if(top_n_peaks is None):
            top_n_peaks = self.peak_idx.shape[0]
        
        if(self.n_dims == 1):
            powers = self.power[self.peak_idx[:,0]]
        elif(self.n_dims == 2):
            powers = self.power[self.peak_idx[:,0], self.peak_idx[:,1]]
        
        ordered_power = torch.argsort(powers, descending=True)
        powers = powers[ordered_power]
        result = torch.zeros([new_points.shape[0], self.n_bands], device=self.device, dtype=torch.float32)
        

        n_peaks = min(self.peak_idx.shape[0], top_n_peaks) if peaks_to_use is None else len(peaks_to_use)
        #print(f"Using top {n_peaks} peaks for reconstruction.")
        for idx in range(n_peaks):
            ind = self.peak_idx[ordered_power[idx] if peaks_to_use is None else ordered_power[peaks_to_use[idx]]]
            if(self.n_dims == 1):
                freq = self.modeled_frequencies[ind[0]]
                theta_MLE = self.wave_coefficients[ind[0]]
                new_points_r = new_points.clone()
            elif(self.n_dims == 2):
                freq = self.modeled_frequencies[ind[1]]
                angle = self.modeled_angles[0][ind[0]]
                theta_MLE = self.wave_coefficients[ind[0], ind[1]]
                R = self.create_rotation_matrix([angle])
                new_points_r = new_points.clone()[:,None,:] @ R[None,...]
            else:
                print("Not implemented yet")
                quit()
            X_fit = self.generate_waves(2*torch.pi*new_points_r[:,0,0]*freq)
            y_fit = (X_fit @ theta_MLE[:,None])
            result += y_fit
        result += self.y_means[None,:]

        return result 

    def transform(self, new_points):
        assert torch.is_tensor(new_points), f"new_points must be a tensor"
        new_points = new_points.to(self.device)

        result = torch.zeros([new_points.shape[0], self.n_bands], device=self.device, dtype=torch.float32)

        for i in range(self.modeled_frequencies.shape[0]):
            if(self.n_dims == 1):
                freq = self.modeled_frequencies[i]
                theta_MLE = self.wave_coefficients[i]
                new_points_r = new_points.clone()
            else:
                print("Not implemented for 2D+")
                quit()

            
            X_fit = self.generate_waves(torch.pi*2*new_points_r * freq)
            y_fit = (X_fit @ theta_MLE[:,None])
            result += y_fit
        result += self.y_means[None,:]

        plt.plot(self.x.cpu().numpy(), self.y.cpu().numpy(), color="blue")
        plt.plot(new_points.cpu().numpy(), result.cpu().numpy(), color="orange")
        plt.show()
        return result 


