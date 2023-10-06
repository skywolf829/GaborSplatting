import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os
from models.LombScargle2D import LombScargle2D

periodic_primitives = load(name='periodic_primitives', 
    sources=[os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda.cpp'), 
            os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda_kernel.cu')])

class PeriodicPrimitivesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gaussian_colors, 
                gaussian_positions, gaussian_scales,
                gaussian_rotations, wave_coefficients,
                num_top_frequencies, num_random_frequencies,
                max_frequency, training=False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if(training):
            coefficients_to_send = torch.empty([wave_coefficients.shape[0],
                                            wave_coefficients.shape[1], 
                                            num_random_frequencies+num_top_frequencies],
                                            device=wave_coefficients.device,
                                            dtype=wave_coefficients.dtype)
            indices_to_send = torch.empty([wave_coefficients.shape[0],
                                            wave_coefficients.shape[1], 
                                            num_random_frequencies+num_top_frequencies],
                                            device=wave_coefficients.device,
                                            dtype=torch.int)
            _, indices = torch.topk(torch.abs(wave_coefficients), num_top_frequencies+num_random_frequencies, dim=2, sorted=False)
            indices_to_send[:,:,0:indices.shape[2]] = indices
            #rand_indices = torch.randint(num_top_frequencies, wave_coefficients.shape[2], 
            #                             [num_random_frequencies],
            #                             dtype=torch.int, device=wave_coefficients.device)
            #indices_to_send[:,:,num_top_frequencies:] = indices[:,:,rand_indices]
            coefficients_to_send = torch.gather(wave_coefficients, 2, indices_to_send.type(torch.long))

        else:
            coefficients_to_send = wave_coefficients
            indices_to_send = torch.arange(0, wave_coefficients.shape[2],
                                           dtype=torch.int, device=wave_coefficients.device)
            indices_to_send = indices_to_send[None,None,:].repeat(
                wave_coefficients.shape[0], wave_coefficients.shape[1], 1)
        # First, get the tip
        outputs = periodic_primitives.forward(x, 
            gaussian_colors, gaussian_positions, gaussian_scales,
            gaussian_rotations, coefficients_to_send, 
            indices_to_send, max_frequency)
        result = outputs[0]

        variables = [x, gaussian_colors, gaussian_positions, 
                    gaussian_scales, gaussian_rotations,
                    wave_coefficients, indices_to_send]
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
                 num_top_freqs = 8, num_random_freqs = 8,
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
        self.wave_coefficients = Parameter(torch.empty([0, num_dimensions, num_frequencies], device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()
    
    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_colors], 'lr': 0.0001, "name": "gaussian_colors"},
            {'params': [self.gaussian_positions], 'lr': 0.0001, "name": "gaussian_positions"},
            {'params': [self.gaussian_scales], 'lr': 0.0001, "name": "gaussian_scales"},
            {'params': [self.gaussian_rotations], 'lr': 0.1, "name": "gaussian_rotations"},
            {'params': [self.wave_coefficients], 'lr': 0.0001, "name": "wave_coefficients"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:           
            total += group['params'][0].numel()
        return total
    
    def add_primitives_random(self, num_gaussians):
        new_colors = 0.1*(torch.randn([num_gaussians, self.n_channels], 
                dtype=torch.float32, device=self.device))
        new_positions = torch.rand([num_gaussians, self.num_dimensions], 
                dtype=torch.float32, device=self.device)
        new_scales = 1 + 0.5*torch.randn([num_gaussians, 2], 
                dtype=torch.float32,  device=self.device)
        new_rotations = torch.pi*torch.rand([num_gaussians, 1],
                dtype=torch.float32,  device=self.device)
        new_wave_coefficients = 0.05*torch.randn([num_gaussians, self.num_dimensions, self.n_frequencies],
                dtype=torch.float32,  device=self.device)
        new_wave_coefficients[:,:,0] += 1.0

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
    
    def add_primitives(self, x, y, n_freqs=256, 
                       max_freq = 128.0, min_influence=1./255.,
                       num_waves=1, num_gaussians=0):
        self.add_primitives_random(num_waves+num_gaussians)
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
                    (torch.linalg.norm(self.wave_coefficients,dim=-1).prod(dim=-1) > min_contribution)
        
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
        decay_loss = 0.0001*torch.abs(self.wave_coefficients)
        final_loss = mse + decay_loss
        losses = {
            "final_loss": final_loss,
            "mse": mse,
            "decay_loss": decay_loss
        }
        return losses, model_out

    def forward(self, x) -> torch.Tensor:
        return PeriodicPrimitivesFunction.apply(x, 
            self.gaussian_colors, self.gaussian_positions, self.gaussian_scales,
            self.gaussian_rotations, self.wave_coefficients, self.num_top_freqs,
            self.num_random_freqs, self.max_frequency, self.training)

    def forward_pytorch(self, x):
        # x is [N, 2]

        if(self.training):
            coefficients_to_send = torch.empty([self.wave_coefficients.shape[0],
                                            self.wave_coefficients.shape[1], 
                                            self.num_random_freqs+self.num_top_freqs],
                                            device=self.device,
                                            dtype=self.wave_coefficients.dtype)
            indices_to_send = torch.empty([self.wave_coefficients.shape[0],
                                            self.wave_coefficients.shape[1], 
                                            self.num_random_freqs+self.num_top_freqs],
                                            device=self.device,
                                            dtype=torch.long)
            _, indices = torch.topk(torch.abs(self.wave_coefficients),
                                     self.num_random_freqs+self.num_top_freqs, dim=2, sorted=False)
            indices_to_send[:,:,0:indices.shape[2]] = indices
            #rand_indices = torch.randint(num_top_frequencies, wave_coefficients.shape[2], 
            #                             [num_random_frequencies],
            #                             dtype=torch.int, device=wave_coefficients.device)
            #indices_to_send[:,:,num_top_frequencies:] = indices[:,:,rand_indices]
            coefficients_to_send = torch.gather(self.wave_coefficients, 2, indices_to_send)

        else:
            coefficients_to_send = self.wave_coefficients
            indices_to_send = torch.arange(0, self.wave_coefficients.shape[2],
                                           dtype=torch.int, device=self.device)
            indices_to_send = indices_to_send[None,None,:].repeat(
                self.wave_coefficients.shape[0], self.wave_coefficients.shape[1], 1)
            
        t_x = x[:,None,0] - self.gaussian_positions[None,:,0]
        t_y = x[:,None,1] - self.gaussian_positions[None,:,1]
        rs_x = self.gaussian_scales[None,:,0]*(t_x*torch.cos(self.gaussian_rotations[None,:,0]) + \
                t_y*torch.sin(self.gaussian_rotations[None,:,0]))
        rs_y = self.gaussian_scales[None,:,1]*(t_x*-torch.sin(self.gaussian_rotations[None,:,0]) + \
                t_y*torch.cos(self.gaussian_rotations[None,:,0]))
        g = torch.exp(-(rs_x*rs_x + rs_y*rs_y)/2.)
        output = g@self.gaussian_colors
            
        return output 