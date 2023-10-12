import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os
import numpy as np

periodic_primitives = load(name='periodic_primitives', 
    sources=[os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda.cpp'), 
            os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitivesCUDA", 'periodic_primitives2DRGB_cuda_kernel.cu')], verbose=False)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class PeriodicPrimitivesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gaussian_colors, 
                gaussian_positions, gaussian_scales,
                gaussian_rotations, topk_wave_coefficients,
                topk_wave_indices,
                num_top_frequencies, num_random_frequencies,
                max_frequency, gaussian_only):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        outputs = periodic_primitives.forward(x, 
            gaussian_colors, gaussian_positions, gaussian_scales,
            gaussian_rotations, topk_wave_coefficients, topk_wave_indices, 
            max_frequency, gaussian_only)
        result = outputs[0]

        variables = [x, gaussian_colors, gaussian_positions, 
                    gaussian_scales, gaussian_rotations,
                    topk_wave_coefficients, topk_wave_indices]
        ctx.save_for_backward(*variables)
        ctx.max_frequency = max_frequency
        ctx.gaussian_only = gaussian_only
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        outputs = periodic_primitives.backward(grad_output, *ctx.saved_tensors, 
                                               ctx.max_frequency, ctx.gaussian_only)
        
        grad_gaussian_colors, grad_gaussian_positions, grad_gaussian_scales, \
            grad_gaussian_rotations, grad_wave_coefficients = outputs
        return grad_output, grad_gaussian_colors, grad_gaussian_positions, \
              grad_gaussian_scales, grad_gaussian_rotations, grad_wave_coefficients, \
              None, None, None, None, None


class PeriodicPrimitives2D(torch.nn.Module):
    def __init__(self, num_dimensions = 2, n_channels = 3, 
                 num_frequencies = 1024, max_frequency = 1024.,
                 num_top_freqs = 4, num_random_freqs = 0,
                 gaussian_only = False,
                 device = "cuda"):
        super(PeriodicPrimitives2D, self).__init__()
        self.device = device
        self.register_buffer("num_dimensions",
            torch.tensor(num_dimensions), persistent=True)
        self.register_buffer("n_channels",
            torch.tensor(n_channels), persistent=True)
        self.register_buffer("num_frequencies",
            torch.tensor(num_frequencies), persistent=True)
        self.register_buffer("max_frequency",
            torch.tensor(max_frequency), persistent=True)
        self.register_buffer("num_top_freqs",
            torch.tensor(num_top_freqs), persistent=True)
        self.register_buffer("num_random_freqs",
            torch.tensor(num_random_freqs), persistent=True)
        self.register_buffer("gaussian_only", 
            torch.tensor(gaussian_only), persistent=True)

        self.gaussian_colors = Parameter(torch.empty([0, n_channels], device=device, dtype=torch.float32)) 
        self.gaussian_positions = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))  
        self.gaussian_scales = Parameter(torch.empty([0, num_dimensions], device=device, dtype=torch.float32))    
        self.gaussian_rotations = Parameter(torch.empty([0, 1], device=device, dtype=torch.float32))      
        self.wave_coefficients = Parameter(torch.empty([0, num_frequencies, num_dimensions], device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()
        self.position_scheduler = get_expon_lr_func(lr_init=0.005,
                                                    lr_final=0.00001,
                                                    max_steps=30000)
        self.scale_scheduler = get_expon_lr_func(lr_init=0.005,
                                                    lr_final=0.00001,
                                                    max_steps=30000)
        self.wave_scheduler = get_expon_lr_func(lr_init=0.005,
                                                    lr_final=0.00001,
                                                    max_steps=30000)
        self.rotation_scheduler = get_expon_lr_func(lr_init=0.005,
                                                    lr_final=0.00001,
                                                    max_steps=30000)
        self.color_scheduler = get_expon_lr_func(lr_init=0.005,
                                                    lr_final=0.00001,
                                                    max_steps=30000)

    def create_optimizer(self):
        l = [
            {'params': [self.gaussian_colors], 'lr': 0.005, "name": "gaussian_colors", "weight_decay": 1e-6},
            {'params': [self.gaussian_positions], 'lr': 0.005, "name": "gaussian_positions"},
            {'params': [self.gaussian_scales], 'lr': 0.005, "name": "gaussian_scales"},
            {'params': [self.gaussian_rotations], 'lr': 0.005, "name": "gaussian_rotations"},
            {'params': [self.wave_coefficients], 'lr': 0.005, "name": "wave_coefficients"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=[0.9, 0.999])
        return optimizer
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if("positions" in param_group['name']):
                param_group['lr'] = self.position_scheduler(iteration)
            if("scale" in param_group['name']):
                param_group['lr'] = self.scale_scheduler(iteration)
            if("rotation" in param_group['name']):
                param_group['lr'] = self.rotation_scheduler(iteration)
            if("color" in param_group['name']):
                param_group['lr'] = self.color_scheduler(iteration)
            if("wave" in param_group['name']):
                param_group['lr'] = self.wave_scheduler(iteration)
        return iteration
            
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:    
            if(group['name'] == 'wave_coefficients' and not self.gaussian_only):
                total += self.gaussian_colors.shape[0]*(self.num_top_freqs + self.num_random_freqs)*self.gaussian_positions.shape[1]
            elif(group['name'] != 'wave_coefficients'):   
                total += group['params'][0].numel()
        return total
    
    def effective_param_count(self):
        total = 0
        for group in self.optimizer.param_groups:    
            if(group['name'] == 'wave_coefficients' and not self.gaussian_only):
                top_k_coeffs, _ = self.get_topk_waves()
                above_thresh = torch.abs(top_k_coeffs) > 1./500.
                total += above_thresh.sum()
            elif(group['name'] != 'wave_coefficients'):   
                total += group['params'][0].numel()
        return total
    
    def get_weighed_frequency_dist(self):
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        frequencies = top_k_indices*self.max_frequency / self.num_frequencies
        return frequencies.flatten()

    def add_primitives_random(self, num_gaussians):
        new_colors = 0.05*(torch.randn([num_gaussians, self.n_channels], 
                dtype=torch.float32, device=self.device))
        new_positions = torch.rand([num_gaussians, self.num_dimensions], 
                dtype=torch.float32, device=self.device)
        new_scales = 2 + 1*torch.randn([num_gaussians, 1], 
                dtype=torch.float32,  device=self.device).repeat(1, 2) + \
                0.05*torch.randn([num_gaussians, 2], dtype=torch.float32,  device=self.device)
        new_rotations = torch.pi*torch.rand([num_gaussians, 1],
                dtype=torch.float32,  device=self.device)
        new_wave_coefficients = 0.1*torch.randn([num_gaussians, self.num_frequencies, self.num_dimensions],
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
    
    def split_prims(self, grads_pos, grads_scales, grads_rotations, num_primitives):
        summed_grads = grads_pos.abs().norm(dim=1) + grads_scales.abs().sum(dim=1) + grads_rotations.abs().sum(dim=1) 
        
        new_prims = min(num_primitives, self.get_num_primitives())
        _, indices = torch.topk(summed_grads, new_prims)

        stds = 1/(2*torch.exp(self.gaussian_scales[indices].clone()))
        #stds = stds*stds
        means = torch.zeros((stds.shape[0], 2), device=self.device, dtype=torch.float32)
        samples = torch.normal(mean=means, std=stds)
        new_rotations = self.gaussian_rotations[indices].clone()
        rotated_samples = torch.stack([samples[:,0]*torch.cos(-new_rotations[:,0]) + samples[:,1]*torch.sin(-new_rotations[:,0]),
                               samples[:,0]*-torch.sin(-new_rotations[:,0]) + samples[:,1]*torch.cos(-new_rotations[:,0])], dim=-1)
        new_positions = self.gaussian_positions[indices].clone() + rotated_samples
        new_colors = self.gaussian_colors[indices].clone() * 0.2
        new_scales = self.gaussian_scales[indices].clone() + np.log(1.8)
        new_wave_coefficients = self.wave_coefficients[indices].clone()

        self.gaussian_scales[indices] += np.log(1.4)
        self.gaussian_colors[indices] *= 0.8

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

        return new_prims
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        with torch.no_grad():
            dict = torch.load(path)
            for name in dict:
                if name == "gaussian_colors":
                    self.gaussian_colors = Parameter(dict[name])
                if name == "gaussian_positions":
                    self.gaussian_positions = Parameter(dict[name])
                if name == "gaussian_scales":
                    self.gaussian_scales = Parameter(dict[name])
                if name == "gaussian_rotations":
                    self.gaussian_rotations = Parameter(dict[name])
                if name == "wave_coefficients":
                    self.wave_coefficients = Parameter(dict[name])
                if name == "num_dimensions":
                    self.num_dimensions = dict[name]
                if name == "n_channels":
                    self.n_channels = dict[name]
                if name == "num_frequencies":
                    self.num_frequencies = dict[name]
                if name == "max_frequency":
                    self.max_frequency = dict[name]
                if name == "num_top_freqs":
                    self.num_top_freqs = dict[name]
                if name == "num_random_freqs":
                    self.num_random_freqs = dict[name]
                if name == "gaussian_only":
                    self.gaussian_only = dict[name]
        print("Successfully loaded trained model.")
    
    def vis_heatmap(self, points):
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        outputs = periodic_primitives.heatmap(points, 
            self.gaussian_positions, torch.exp(self.gaussian_scales),
            self.gaussian_rotations, 
            top_k_coeffs, top_k_indices, 
            self.max_frequency, self.gaussian_only)
        heatmap = outputs[0]
        #heatmap -= heatmap.min()
        heatmap = heatmap / heatmap.max()
        return heatmap

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

    def prune_primitives(self, min_contribution:int=1./1000.):

        exp_scales = torch.exp(self.gaussian_scales)
        r = 3/exp_scales.min(dim=1).values
        on_image = (self.gaussian_positions[:,0]+r >= 0) * (self.gaussian_positions[:,0]-r <= 1.0) * \
                    (self.gaussian_positions[:,1]+r >= 0) * (self.gaussian_positions[:,1]-r <= 1.0)
        gaussians_mask = (torch.linalg.norm(self.gaussian_colors,dim=-1) > min_contribution) * on_image
        if(not self.gaussian_only):
            gaussians_mask *= (torch.linalg.norm(self.wave_coefficients,dim=1).prod(dim=-1) > min_contribution)

        to_remove = 0
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
        return to_remove
        
    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        final_loss = mse
        losses = {
            "final_loss": final_loss,
            "mse": mse,
        }
        return losses, model_out

    def get_topk_waves(self):
        if(self.training):
            coefficients_to_send, indices_to_send = torch.topk(torch.abs(self.wave_coefficients),
                                    self.num_top_freqs, dim=1, sorted=False)
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
            self.num_top_freqs, self.num_random_freqs, self.max_frequency, self.gaussian_only)

    def forward_pytorch(self, x):
        # x is [N, 2]
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        t_x = x[:,None,0] - self.gaussian_positions[None,:,0]
        t_y = x[:,None,1] - self.gaussian_positions[None,:,1]
        cosr = torch.cos(self.gaussian_rotations[:,0])[None,...]
        sinr = torch.sin(self.gaussian_rotations[:,0])[None,...]
        rs_x = torch.exp(self.gaussian_scales[None,:,0])*(t_x*cosr  + t_y*sinr)
        rs_y = torch.exp(self.gaussian_scales[None,:,1])*(t_x*-sinr + t_y*cosr)
        g = torch.exp(-0.5*(rs_x*rs_x + rs_y*rs_y))
        if(not self.gaussian_only):
            w = top_k_coeffs[None,...]*torch.cos(torch.stack([t_x, t_y], dim=-1)[:,:,None,:]*self.max_frequency*top_k_indices[None,...]/self.n_frequencies)
            w = w.sum(dim=2).prod(dim=2)
            g = g*w
        return g@self.gaussian_colors 