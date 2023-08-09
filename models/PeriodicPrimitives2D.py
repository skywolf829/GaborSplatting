import torch
from torch.nn.parameter import Parameter
from models.LombScargle import MyLombScargleModel

class PeriodicPrimitives2D(torch.nn.Module):

    def __init__(self, n_gaussians, device="cuda"):
        super().__init__()
        print(f"Initializing model with {n_gaussians} waves.")
        # Parameters
        self.freqs = Parameter(1024*torch.rand([n_gaussians, 1], dtype=torch.float32, device=device))
        self.rotations = Parameter(torch.pi*torch.rand([n_gaussians, 1], dtype=torch.float32, device=device))
        self.coeffs = Parameter(((1/n_gaussians)**0.5)*torch.rand([n_gaussians, 2], dtype=torch.float32, device=device))

    def init_lombscargle(self, x, y):
        print(f"Initializing Lomb-Scargle model on training data...")
        self.ls_model = MyLombScargleModel(x, y, self.freqs.device)
        wavelengths = torch.linspace(1./1024., 1.0, 1024)
        freqs = (1./wavelengths).flip([0])
        self.ls_model.fit(freqs, 
                          angles=[torch.linspace(0., torch.pi, 180)], 
                          linear_in_frequency=False)
        self.data_periodogram = self.ls_model.get_power()
        self.ls_model.find_peaks()
        target_freqsangles = self.ls_model.get_peak_freq_angles()
        target_coeffs = self.ls_model.get_peak_coeffs()
        self.freqs = Parameter(target_freqsangles[:,0:1])
        self.rotations = Parameter(target_freqsangles[:,1:2])
        self.coeffs = Parameter(target_coeffs)

    def create_optimizer(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001, 
            betas=[0.9, 0.99])
        return optim
    
    def create_rotation_matrices(self):
        return torch.stack([torch.cos(self.rotations), -torch.sin(self.rotations),
                             torch.sin(self.rotations), torch.cos(self.rotations)], dim=-1).reshape(-1, 2, 2)

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        l1 = torch.nn.functional.l1_loss(model_out,y)
        return l1, model_out

    def forward(self, x):
        # x is [N, 2]
        R = self.create_rotation_matrices() # [n_gaussians, 2, 2]
        rel_x = x[:,None,None,:] @ R # [N, n_gaussians, 1, 2]
        # [N n_gaussians, 1]
        vals =  self.coeffs[None,:,0:1]*torch.sin(2*torch.pi*rel_x[:,:,:,0]*self.freqs[None,...]) + \
                self.coeffs[None,:,1:2]*torch.cos(2*torch.pi*rel_x[:,:,:,0]*self.freqs[None,...])
        vals = self.ls_model.get_offsets()[None,...] + vals.sum(dim=1)
        return vals # [N, channels]

