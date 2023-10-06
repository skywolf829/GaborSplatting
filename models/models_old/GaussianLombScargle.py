import torch
import matplotlib.pyplot as plt
import os
from typing import List
from time import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cuda.matmul.allow_tf32 = True

class GaussianLombScargleModel():
    def __init__(self, x: torch.Tensor, y:torch.Tensor, device="cuda"):
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
        self.y_means : torch.Tensor = y.mean(dim=0).to(device)

        self.n_items : int = x.shape[0]
        self.n_dims : int = x.shape[1]
        self.n_bands : int = y.shape[1]
        self.unique_bands = torch.arange(self.n_bands, device=device)

        self.modeled_frequencies : torch.Tensor = None
        self.modeled_angles : torch.Tensor = None

        self.wave_coefficients : torch.Tensor = None
        self.power : torch.Tensor = None

        self.peak_idx : torch.Tensor = None

        self.linear_in_frequency : bool = False
        self.perfect_fit : bool = False
        
        self.nterms_base = 1
        self.nterms_band = 1
        self.reg_band = 1e-6

    def create_rotation_matrix(self, theta : torch.Tensor):
        """
        Creates 2D or 3D rotation matrix
        """
        for i in range(len(theta)):
            assert torch.is_tensor(theta[i]), f"theta[{i}] must be a tensor"
            assert theta[i].ndim == 0, f"theta must be 0 dimensional, found {theta[i].ndim}"

        if len(theta) == 1:
            return torch.tensor([
                [torch.cos(theta[0]), -torch.sin(theta[0])],
                [torch.sin(theta[0]), torch.cos(theta[0])]
            ], device = self.device, dtype=torch.float32)
        elif len(theta) == 2:
            print("Not yet implemented")
            quit(0)
        else:
            print(f"Not supported")

    def generate_waves(self, v):
        return torch.stack([torch.sin(v), torch.cos(v), torch.ones_like(v)], dim=-1)

    def fit_angle(self, angles, idx, chi2, yw):

        # Rotate coordinates if 2D or 3D
        if(self.n_dims > 1):
            angle = [angles[j][idx[j]] for j in range(len(idx))]
            R = self.create_rotation_matrix(angle)
            x_r = self.x[:,None,:] @ R[None,...]
            x_r = x_r.squeeze()[:,0]
        elif self.n_dims == 1:
            x_r = self.x[:,0]
        
        # Construct X - design matrix of the stationary sinusoid model
        # Most time is spent making this.
        X = self.generate_waves(2*torch.pi*x_r[None,:] * self.modeled_frequencies[:,None])
        XTX = X.mT @ X
        XTy = X.mT @ yw[None,...]
        
        # Matrix Algebra to calculate the Lomb-Scargle power at each omega step
        theta_MLE = torch.linalg.solve(XTX, XTy)

        # update model
        self.wave_coefficients[idx] = theta_MLE.squeeze()

        # Correct way
        self.power[idx] = ((XTy.mT @ theta_MLE) / chi2).squeeze()
        # better quality for model
        #self.power[idx] = torch.linalg.norm(theta_MLE.squeeze(),dim=-1)

    def fit_angle_perfect(self, angles, idx, chi2, yw):

        # Rotate coordinates if 2D or 3D
        if(self.n_dims > 1):
            angle = [angles[j][idx[j]] for j in range(len(idx))]
            R = self.create_rotation_matrix(angle)
            x_r = self.x[:,None,:] @ R[None,...]
            x_r = x_r.squeeze()[:,0]
        else:
            x_r = self.x[:,0]
        X = []
        for j in range(self.modeled_frequencies.shape[0]):   
            #print(f"[angle {i}/{len(it)}, freq {j}/{frequencies.shape[0]}]")                  
            # Construct X - design matrix of the stationary sinusoid model
            X.append(self.generate_waves(2*torch.pi*x_r*self.modeled_frequencies[j]))
        
        X = torch.concat(X, dim=-1)
        XTX = X.t() @ X        
        XTy = X.t() @ yw

        # Matrix Algebra to calculate the Lomb-Scargle power at each omega step
        try:
            theta_MLE = torch.linalg.solve(XTX, XTy)
        # If X'X is not invertible, use pseudoinverse instead
        except torch.linalg.LinAlgError:
            theta_MLE = torch.linalg.lstsq(XTX, XTy, rcond=None)[0]
        # update model
        
        self.wave_coefficients[idx] = theta_MLE.squeeze().reshape(-1, 3)
        self.power[idx] = (XTy.reshape(-1, 3) @ theta_MLE.reshape(-1, 3).t()).sum(dim=-1) / chi2

    def fit(self, frequencies:torch.Tensor, angles: List[torch.Tensor] = None, linear_in_frequency : bool=False, perfect_fit: bool = False):
        """
        Fits the model to each frequency in frequencies.
        Also fits angles, which are given as a list of angles in theta, phi order.
        """
        
        assert torch.is_tensor(frequencies), f"frequencies is not a tensor"
        assert frequencies.ndim == 1, f"frequencies should only have 1 dim, found {frequencies.shape}"

        if angles is None and self.n_dims > 1:
            for i in range(self.n_dims - 1):
                angles.append(torch.tensor([0], device=self.device))
        if angles is not None:
            for i in range(len(angles)):
                assert torch.is_tensor(angles[i]), f"angles[{i}] is not a tensor"
                assert angles[i].ndim == 1, \
                    f"angles has ndim = {angles.ndim}, but should be 1"
            assert self.n_dims == 1 or len(angles) == self.n_dims - 1, \
                f"Given data that is {self.n_dims}D, expected {self.n_dims-1} entries in angles, found {len(angles)}"

        self.perfect_fit = perfect_fit

        angles_shapes = []
        if(angles is not None):
            for i in range(len(angles)):
                angles[i] = angles[i].to(self.device)
                angles_shapes.append(angles[i].shape[0])

        self.linear_in_frequency = linear_in_frequency
        self.peak_idx = None
        
        frequencies = frequencies.to(self.device)

        self.modeled_frequencies = frequencies
        self.modeled_angles = angles
        self.wave_coefficients = torch.empty([*angles_shapes, frequencies.shape[0], 3],
                                              device=self.device, dtype=torch.float32)
        self.power = torch.empty([*angles_shapes, frequencies.shape[0]], 
                                 device = self.device, dtype=torch.float32)

        # Construct weighted y matrix by subtracting means for each band
        yw = self.y #- self.y_means

        # Calculate chi-squared
        chi2 = yw.t() @ yw  # reference chi2 for later comparison

        # Create iterator for angles
        if(self.n_dims == 1):
            it = [()]
        elif(self.n_dims == 2):
            it = [(idx,) for idx in range(angles_shapes[0])]
        elif(self.n_dims == 3):
            it = [(idx_t, idx_p) for idx_t in range(angles_shapes[0]) for idx_p in range(angles_shapes[1])]

        fn = self.fit_angle
        if(self.perfect_fit):
            fn = self.fit_angle_perfect
        start_time = time()        
        with torch.no_grad():
            for i in range(len(it)):  
                fn(angles, it[i], chi2, yw)
        end_time = time()
        #print(f"Fitting took {end_time-start_time:0.02f} sec")

    def get_next_gaussian_peak(self, frequencies, angles,linear_in_frequency : bool=False):
        """
        Returns the next gaussian peak 
        """
        
        assert torch.is_tensor(frequencies), f"frequencies is not a tensor"
        assert frequencies.ndim == 1, f"frequencies should only have 1 dim, found {frequencies.shape}"

        if angles is None and self.n_dims > 1:
            for i in range(self.n_dims - 1):
                angles.append(torch.tensor([0], device=self.device))
        if angles is not None:
            for i in range(len(angles)):
                assert torch.is_tensor(angles[i]), f"angles[{i}] is not a tensor"
                assert angles[i].ndim == 1, \
                    f"angles has ndim = {angles.ndim}, but should be 1"
            assert self.n_dims == 1 or len(angles) == self.n_dims - 1, \
                f"Given data that is {self.n_dims}D, expected {self.n_dims-1} entries in angles, found {len(angles)}"

        angles_shapes = []
        if(angles is not None):
            for i in range(len(angles)):
                angles[i] = angles[i].to(self.device)
                angles_shapes.append(angles[i].shape[0])

        frequencies = frequencies.to(self.device)
        self.linear_in_frequency = linear_in_frequency
        self.modeled_frequencies = frequencies
        self.modeled_angles = angles

        # Create iterator for angles
        if(self.n_dims == 1):
            it = [()]
        elif(self.n_dims == 2):
            it = [(idx,) for idx in range(angles_shapes[0])]
        elif(self.n_dims == 3):
            it = [(idx_t, idx_p) for idx_t in range(angles_shapes[0]) for idx_p in range(angles_shapes[1])]

        self.peak_idx = None      
        self.wave_coefficients = torch.empty([*angles_shapes, frequencies.shape[0], 3],
                                              device=self.device, dtype=torch.float32)
        self.power = torch.empty([*angles_shapes, frequencies.shape[0]], 
                                 device = self.device, dtype=torch.float32)
        yw = self.y #- self.y_means
        chi2 = yw.t() @ yw

        with torch.no_grad():
            for i in range(len(it)):  
                self.fit_angle(angles, it[i], chi2, yw)
                
        self.find_peaks(top_n = 1)
        #self.plot_power()
        first_coeffs = self.get_peak_coeffs()[:,0:2]
        first_signs = self.get_peak_coeffs()[:,2]
        first_waves = 1/self.get_peak_freq_angles()[:,0]
        first_rotation = self.get_peak_freq_angles()[:,1]

        y_new = self.y - self.transform_from_peaks(self.x)
        yw = y_new
        chi2 = yw.t() @ yw


        self.peak_idx = None      
        self.wave_coefficients = torch.empty([*angles_shapes, frequencies.shape[0], 3],
                                              device=self.device, dtype=torch.float32)
        self.power = torch.empty([*angles_shapes, frequencies.shape[0]], 
                                 device = self.device, dtype=torch.float32)

        
        with torch.no_grad():
            for i in range(len(it)):  
                self.fit_angle(angles, it[i], chi2, yw)
        
        self.find_peaks(top_n = 1)
        #self.plot_power()
        second_coeffs = self.get_peak_coeffs()[:,0:2]
        second_signs = self.get_peak_coeffs()[:,2]
        second_waves = 1/self.get_peak_freq_angles()[:,0]
        second_rotation = self.get_peak_freq_angles()[:,1]

        coeffs = torch.cat([first_coeffs, second_coeffs], dim=0)
        signs = torch.cat([first_signs, second_signs], dim=0)
        waves = torch.cat([first_waves, second_waves], dim=0)
        rotations = torch.cat([first_rotation, second_rotation], dim=0)

        return coeffs, signs, waves, rotations


        


    def get_power(self):
        assert self.power is not None, f"power is none, fit a model first"
        return self.power
    
    def get_wave_coefficients(self):
        assert self.wave_coefficients is not None, f"wave_coefficients is none, fit a model first"
        return self.wave_coefficients
    
    def get_offsets(self):
        assert self.y_means is not None, f"offsets is none, fit a model first"
        return self.y_means
    
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
        criterion = torch.linalg.norm(self.wave_coefficients[:,:,0:2],dim=-1) > min_influence
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
            
        #print(f"Extracted {self.peak_idx.shape[0]} peaks.")
        return self.peak_idx

    def get_peak_freq_angles(self):
        assert self.peak_idx is not None, "Must compute peaks first with find_peaks()"

        if(self.n_dims == 1):
            freqs = self.modeled_frequencies[self.peak_idx[:,0]]
            return freqs
        elif(self.n_dims == 2):
            freqs = self.modeled_frequencies[self.peak_idx[:,1]]
            angles = self.modeled_angles[0][self.peak_idx[:,0]]
            return torch.stack([freqs, angles], dim=-1)

    def get_peak_coeffs(self):
        assert self.peak_idx is not None, "Must compute peaks first with find_peaks()"
        coeffs = self.wave_coefficients[self.peak_idx[:,0], self.peak_idx[:,1],:]
        return coeffs

    def get_num_peaks(self):
        assert self.peak_idx is not None, f"Must find peaks first"
        return self.peak_idx.shape[0]

    def plot_power(self):
        assert self.n_dims == 1 or self.modeled_angles is not None, f"Need modeled angles"
        assert self.modeled_frequencies is not None, f"Need modeled frequencies"
        assert self.power is not None, f"Requires computed powers"

        if(self.n_dims == 1):
            x_axis = self.modeled_frequencies.clone()
            values = self.power.clone()
            x_label = "Frequencies (Hz)"
            y_label = "Power"
            if not self.linear_in_frequency:
                x_axis = torch.flip(1/x_axis, dims=[0])
                values = torch.flip(values, dims=[0])
                x_label = "Wavelength"
            plt.plot(x_axis.cpu().numpy(), values.cpu().numpy())

            if(self.peak_idx is not None):
                idx_peaks = self.peak_idx.clone()
                if(not self.linear_in_frequency):
                    idx_peaks[:,0] = x_axis.shape[0] - 1 - idx_peaks[:,0]
                x = x_axis[idx_peaks[:,0]]
                y = values[idx_peaks[:,0]]
                plt.scatter(x.cpu().numpy(), y.cpu().numpy())

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.show()


        elif(self.n_dims == 2):
            x_axis = self.modeled_frequencies.clone()
            y_axis = self.modeled_angles[0].clone()
            values = self.power.clone()
            y_label = "Angles (rad)"
            x_label = "Frequencies (Hz)"
            if not self.linear_in_frequency:
                x_axis = torch.flip(1/x_axis, dims=[0])
                values = torch.fliplr(values)
                x_label = "Wavelength"
            
            x_min = x_axis[0].cpu().numpy()
            x_max = x_axis[-1].cpu().numpy()
            y_min = y_axis[0].cpu().numpy()
            y_max = y_axis[-1].cpu().numpy()
            x_width = (x_max-x_min)
            y_height = (y_max-y_min)

            plt.imshow(values.cpu().numpy(), extent=[x_min, x_max, y_min, y_height], 
               aspect=x_width/y_height)

            if(self.peak_idx is not None):
                idx_peaks = self.peak_idx.clone().cpu()
                idx_peaks[:,0] = y_axis.shape[0] - 1 - idx_peaks[:,0]
                if(not self.linear_in_frequency):
                    idx_peaks[:,1] = x_axis.shape[0] - 1 - idx_peaks[:,1]
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


