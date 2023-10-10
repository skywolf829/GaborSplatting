import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os

periodic_primitives = load(name='periodic_primitives', 
    sources=[os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda.cpp'), 
            os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda_kernel.cu')], verbose=False)

class PeriodicPrimitivesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gaussian_colors, 
                gaussian_positions, gaussian_scales,
                gaussian_rotations, topk_wave_coefficients,
                topk_wave_indices,
                num_top_frequencies, num_random_frequencies,
                max_frequency):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        outputs = periodic_primitives.forward(x, 
            gaussian_colors, gaussian_positions, gaussian_scales,
            gaussian_rotations, topk_wave_coefficients, topk_wave_indices, 
            max_frequency)
        result = outputs[0]

        variables = [x, gaussian_colors, gaussian_positions, 
                    gaussian_scales, gaussian_rotations,
                    topk_wave_coefficients, topk_wave_indices]
        ctx.save_for_backward(*variables)
        ctx.max_frequency = max_frequency
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        outputs = periodic_primitives.backward(grad_output, *ctx.saved_tensors, ctx.max_frequency)
        
        grad_gaussian_colors, grad_gaussian_positions, grad_gaussian_scales, \
            grad_gaussian_rotations, grad_wave_coefficients = outputs
        return grad_output, grad_gaussian_colors, grad_gaussian_positions, \
              grad_gaussian_scales, grad_gaussian_rotations, grad_wave_coefficients, \
              None, None, None, None


class PeriodicPrimitives2D(torch.nn.Module):
    def __init__(self, num_dimensions = 2, n_channels = 3, 
                 num_frequencies = 1024, max_frequency = 1024.,
                 num_top_freqs = 2, num_random_freqs = 2,
                 device = "cuda"):
        super(PeriodicPrimitives2D, self).__init__()
        self.device = device
        self.num_dimensions = num_dimensions
        self.n_channels = n_channels
        self.n_frequencies = num_frequencies
        self.max_frequency = max_frequency
        self.num_top_freqs = num_top_freqs
        self.num_random_freqs = num_random_freqs

        self.gaussian_colors = Parameter(torch.empty([0, n_channels], device=device, dtype=torch.float32)) 
        self.gaussian_positions = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))  
        self.gaussian_scales = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))    
        self.gaussian_rotations = Parameter(torch.empty([0, 1], device=device, dtype=torch.float32))      
        self.wave_coefficients = Parameter(torch.empty([0, num_frequencies, num_dimensions], device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()
    
    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_colors], 'lr': 0.01, "name": "gaussian_colors"},
            {'params': [self.gaussian_positions], 'lr': 0.01, "name": "gaussian_positions"},
            {'params': [self.gaussian_scales], 'lr': 0.01, "name": "gaussian_scales"},
            {'params': [self.gaussian_rotations], 'lr': 0.1, "name": "gaussian_rotations"},
            {'params': [self.wave_coefficients], 'lr': 0.01, "name": "wave_coefficients"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:    
            if(group['name'] == 'wave_coefficients'):
                total += self.gaussian_colors.shape[0]*(self.num_top_freqs + self.num_random_freqs)*2
            else:       
                total += group['params'][0].numel()
        return total
    
    def add_primitives_random(self, num_gaussians):
        new_colors = 0.1*(torch.randn([num_gaussians, self.n_channels], 
                dtype=torch.float32, device=self.device))
        new_positions = torch.rand([num_gaussians, self.num_dimensions], 
                dtype=torch.float32, device=self.device)
        new_scales = 4 - 2*torch.randn([num_gaussians, 2], 
                dtype=torch.float32,  device=self.device)
        new_rotations = torch.pi*torch.rand([num_gaussians, 1],
                dtype=torch.float32,  device=self.device)
        new_wave_coefficients = 0.05*torch.randn([num_gaussians, self.n_frequencies, self.num_dimensions],
                dtype=torch.float32,  device=self.device)
        new_wave_coefficients[:,0,:] += 1.0

        tensor_dict = {
            "gaussian_colors": new_colors, 
            "gaussian_positions": new_positions,
            "gaussian_scales": new_scales,
            "gaussian_rotations": new_rotations,
            "wave_coefficients": new_wave_coefficients
        }

        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.gaussian_colors = updated_params['gaussian_colors']
        self.gaussian_positions = updated_params['gaussian_positions']
        self.gaussian_scales = updated_params['gaussian_scales']
        self.gaussian_rotations = updated_params['gaussian_rotations']
        self.wave_coefficients = updated_params['wave_coefficients']

    def get_num_primitives(self):
        return self.gaussian_colors.shape[0]
    
    def add_primitives(self, num_primitives):
        self.add_primitives_random(num_primitives)
        return 0
    
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

        gaussians_mask = (torch.linalg.norm(self.gaussian_colors,dim=-1) > min_contribution) * \
                    (torch.linalg.norm(self.wave_coefficients,dim=1).prod(dim=-1) > min_contribution)
        
        if(len(gaussians_mask.shape)>0):
            to_remove = gaussians_mask.shape[0]-gaussians_mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                updated_params = self.prune_tensors_from_optimizer(gaussians_mask)
                self.gaussian_colors = updated_params['gaussian_colors']
                self.gaussian_positions = updated_params['gaussian_positions']
                self.gaussian_scales = updated_params['gaussian_scales']
                self.gaussian_rotations = updated_params['gaussian_rotations']
                self.wave_coefficients = updated_params['wave_coefficients']
        
    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        decay_loss = 0.0001*torch.abs(self.wave_coefficients).mean()
        final_loss = mse + decay_loss
        losses = {
            "final_loss": final_loss,
            "mse": mse,
            "decay_loss": decay_loss
        }
        return losses, model_out

    def get_topk_waves(self):
        """return torch.empty([self.wave_coefficients.shape[0],
                                            self.wave_coefficients.shape[1], 
                                            self.num_random_freqs+self.num_top_freqs],
                                            device=self.device,
                                            dtype=self.wave_coefficients.dtype), \
            torch.empty([self.wave_coefficients.shape[0],
                                            self.wave_coefficients.shape[1], 
                                            self.num_random_freqs+self.num_top_freqs],
                                            device=self.device,
                                            dtype=torch.int)
        """
        if(self.training):
            coefficients_to_send, indices_to_send = torch.topk(torch.abs(self.wave_coefficients),
                                     self.num_random_freqs+self.num_top_freqs, dim=1, sorted=False)
            coefficients_to_send = coefficients_to_send * torch.gather(self.wave_coefficients.sign(), dim=1, index=indices_to_send)
            
            indices_to_send = indices_to_send.type(torch.int)

        else:
            coefficients_to_send = self.wave_coefficients
            indices_to_send = torch.arange(0, self.wave_coefficients.shape[1],
                                           dtype=torch.int, device=self.device)
            indices_to_send = indices_to_send[None,:,None].repeat(
                self.wave_coefficients.shape[0], 1, self.wave_coefficients.shape[2])
        
        return coefficients_to_send, indices_to_send

    def forward(self, x) -> torch.Tensor:
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        return PeriodicPrimitivesFunction.apply(x, 
            self.gaussian_colors, self.gaussian_positions, torch.exp(self.gaussian_scales),
            self.gaussian_rotations, top_k_coeffs, top_k_indices,
            self.num_top_freqs, self.num_random_freqs, self.max_frequency)

    def forward_pytorch(self, x):
        # x is [N, 2]
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        t_x = x[:,None,0] - self.gaussian_positions[None,:,0]
        t_y = x[:,None,1] - self.gaussian_positions[None,:,1]
        cosr = torch.cos(self.gaussian_rotations[:,0])[None,...]
        sinr = torch.sin(self.gaussian_rotations[:,0])[None,...]
        rs_x = self.gaussian_scales[None,:,0]*(t_x*cosr  + t_y*sinr)
        rs_y = self.gaussian_scales[None,:,1]*(t_x*-sinr + t_y*cosr)
        g = torch.exp(-0.5*(rs_x*rs_x + rs_y*rs_y))
        w = top_k_coeffs[None,...]*torch.cos(x[:,None,None,:]*self.max_frequency*top_k_indices[None,...]/self.n_frequencies)
        w = w.sum(dim=2).prod(dim=2)
        return (g*w)@self.gaussian_colors 