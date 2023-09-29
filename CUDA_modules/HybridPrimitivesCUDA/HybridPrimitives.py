import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os

hybrid_primitives = load(name='hybrid_primitives', 
                    sources=[os.path.join("/".join(__file__.split('/')[0:-1]), 'hybrid_primitives_cuda.cpp'), 
                             os.path.join("/".join(__file__.split('/')[0:-1]), 'hybrid_primitives_cuda_kernel.cu')])

class HybridPrimitivesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gaussian_colors, gaussian_means, gaussian_mats,
            wave_colors, wave_support_means, wave_support_mats, 
            wave_frequencies, wave_coefficients):
        outputs = hybrid_primitives.forward(x, 
            gaussian_colors, gaussian_means, gaussian_mats,
            wave_colors, wave_support_means, wave_support_mats, 
            wave_frequencies, wave_coefficients)
        result = outputs[0]
        #variables = outputs[1:] + [weights]
        #ctx.save_for_backward(*variables)
        return result

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = hybrid_primitives.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class HybridPrimitives(torch.nn.Module):
    def __init__(self, num_dimensions=2, num_channels=3, device="cuda"):
        super(HybridPrimitives, self).__init__()
        self.device = device
        self.num_dimensions = num_dimensions
        self.num_channels = num_channels

        self.gaussian_colors = Parameter(torch.empty([0, num_channels], device=device, dtype=torch.float32)) 
        self.gaussian_means = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))        
        self.gaussian_mats = Parameter(torch.empty([0, num_dimensions, num_dimensions], device=device, dtype=torch.float32))
        
        self.wave_colors = Parameter(torch.empty([0, num_channels], device=device, dtype=torch.float32))
        self.wave_support_means = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))        
        self.wave_support_mats = Parameter(torch.empty([0, num_dimensions, num_dimensions], device=device, dtype=torch.float32))
        self.wave_frequencies = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))    
        self.wave_coefficients = Parameter(torch.empty([0, 5], device=device, dtype=torch.float32))

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
    
    def add_gaussians(self, num_gaussians):
        new_colors = 0.2*(torch.randn([num_gaussians, self.num_channels], 
                dtype=torch.float32, device=self.device))
        new_means = torch.rand([num_gaussians, self.num_dimensions], 
                dtype=torch.float32, device=self.device)
        new_mats = torch.eye(self.num_dimensions, device=self.device, 
                dtype=torch.float32)[None,...].repeat(num_gaussians, 1, 1) * (num_gaussians**2)
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

    def add_wave(self, num_waves):
        new_colors = 0.2*(torch.randn([num_waves, self.num_channels], 
                dtype=torch.float32, device=self.device))
        new_means = torch.rand([num_waves, self.num_dimensions], 
                dtype=torch.float32, device=self.device)
        new_mats = torch.eye(self.num_dimensions, device=self.device, 
                dtype=torch.float32)[None,...].repeat(num_waves, 1, 1) * (num_waves**2)
        new_mats += torch.randn_like(new_mats)*0.1
        new_frequencies = torch.rand([num_waves, self.num_dimensions],
                dtype=torch.float32, device=self.device)
        new_coefficients = torch.rand([num_waves, 5],
                dtype=torch.float32, device=self.device)

        tensor_dict = {
            "wave_colors": new_colors, 
            "wave_support_means": new_means,
            "wave_support_mats": new_mats,
            "wave_frequencies": new_frequencies,
            "wave_coefficients": new_coefficients,
        }

        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.wave_colors = updated_params['wave_colors']
        self.wave_support_means = updated_params['wave_support_means']
        self.wave_support_mats = updated_params['wave_support_mats']
        self.wave_frequencies = updated_params['wave_frequencies']
        self.wave_coefficients = updated_params['wave_coefficients']

    def get_num_waves(self):
        return self.wave_colors.shape[0]
    
    def get_num_gaussians(self):
        return self.gaussian_colors.shape[0]
    
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
    
    def prune_tensors_from_optimizer(self, mask, type):
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

    def forward(self, x):
        return HybridPrimitivesFunction.apply(x, 
            self.gaussian_colors, self.gaussian_means, self.gaussian_mats,
            self.wave_colors, self.wave_support_means, self.wave_support_mats, 
            self.wave_frequencies, self.wave_coefficients)

    def forward_pytorch(self, x):
        # x is [N, 2]
        # gaussian coeffs
        output = torch.zeros([x.shape[0], self.num_channels],
                             dtype=torch.float32, device=self.device)

        # Contributions from waves
        if(self.get_num_waves() > 0):
            # find distance from each query point to each gaussian (expensive)
            rel_x = x[:,None,:] - self.wave_support_means[None,...] # [N, n_gaussians, 2]
            
            # Transform each query by the supporting gassians covariance matrix
            # **10 for the harder edges (square-like)
            # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
            transformed_x = ((rel_x[...,None].mT @ self.wave_support_mats[None,...])**10).sum(dim=-1, keepdim=True)

            # Exponential support
            # [N, n_gaussians, 1, 1]
            gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
            
            # Fitted sinusoidal wave coefficients and frequencies
            # [N, channels]            
            sx = torch.sin(2*torch.pi*x[:,None,0]*self.wave_frequencies[None,...,0])
            cx = torch.cos(2*torch.pi*x[:,None,0]*self.wave_frequencies[None,...,0])
            sy = torch.sin(2*torch.pi*x[:,None,1]*self.wave_frequencies[None,...,1])
            cy = torch.cos(2*torch.pi*x[:,None,1]*self.wave_frequencies[None,...,1])
            vals = self.wave_coefficients[None,:,0]*cx*cy + \
                    self.wave_coefficients[None,:,1]*cx*sy + \
                    self.wave_coefficients[None,:,2]*sx*cy + \
                    self.wave_coefficients[None,:,3]*sx*sy + \
                    self.wave_coefficients[None,:,4]
            
            # Sum together outputs from each gaussian for each point
            # [N, n_primitives, 1]
            vals = gauss_vals*vals[...,None]*self.wave_colors[None,...]
            output = output + vals.sum(dim=1)

        # Contribution from gaussians
        if(self.get_num_gaussians() > 0):
            rel_x = x[:,None,:] - self.gaussian_means[None,...] # [N, n_gaussians, 2]
            transformed_x = ((rel_x[...,None].mT @ self.gaussian_mats[None,...])**2).sum(dim=-1, keepdim=True)
            gauss_vals = torch.exp(-(transformed_x[:,:,0])/2)
            gauss_vals = gauss_vals*self.gaussian_colors[None,...]
            output = output + gauss_vals.sum(dim=1)
            
        return output 