import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os
import numpy as np

periodic_primitives = load(name='periodic_primitives', 
    sources=[os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitives3D", 'periodic_primitives.cpp'), 
            os.path.join(os.sep.join(__file__.split(os.sep)[0:-1]),"..", "CUDA_Modules", 
                            "PeriodicPrimitives3D", 'periodic_primitives_cuda_kernel.cu')], verbose=False)

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
            delay_rate = 0.
            # A kind of reverse cosine decay.
            #delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            #    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            #)
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class PeriodicPrimitivesFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, colors, opacities, background_color,
                positions, scales, scale_modifier,
                rotations, 
                topk_wave_coefficients, topk_wave_indices,
                max_frequency, 
                cam_position, view_matrix, VP_matrix,
                fov_x, fov_y, 
                image_width, image_height,
                gaussian_only, heatmap=False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        outputs = periodic_primitives.forward(colors, 
                opacities, background_color,
                positions, scales, scale_modifier,
                rotations, 
                topk_wave_coefficients, topk_wave_indices,
                max_frequency, 
                cam_position, view_matrix, VP_matrix,
                fov_x, fov_y, 
                image_width, image_height,
                gaussian_only, heatmap)
        
        result, final_T, num_contributors = outputs
        """
        variables = [colors, 
                opacities, background_color,
                positions, scales, 
                rotations, 
                topk_wave_coefficients, topk_wave_indices,
                cam_position, view_matrix, VP_matrix]
        
        ctx.save_for_backward(*variables)
        ctx.max_frequency = max_frequency
        ctx.fov_x = fov_x
        ctx.fov_y = fov_y
        ctx.scale_modifier = scale_modifier
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.gaussian_only = gaussian_only
        ctx.sorted_primitive_indices = sorted_primitives_indices
        ctx.blocks_start_end_indices = blocks_start_end_indices
        """
        return result
    

class PeriodicPrimitives3D(torch.nn.Module):
    def __init__(self, opt):
        super(PeriodicPrimitives3D, self).__init__()
        device = opt['device']
        self.opt = opt

        self.primitive_alphas = Parameter(torch.empty([0, 1], device=device, dtype=torch.float32)) 
        self.primitive_colors = Parameter(torch.empty([0, 3], device=device, dtype=torch.float32)) 
        self.primitive_positions = Parameter(torch.empty([0, 3], device=device, dtype=torch.float32))  
        self.primitive_scales = Parameter(torch.empty([0, 3], device=device, dtype=torch.float32))    
        self.primitive_rotations = Parameter(torch.empty([0, 4], device=device, dtype=torch.float32))      
        self.wave_coefficients = Parameter(torch.empty([0, opt['num_total_frequencies']], device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()
        
        self.alpha_scheduler = get_expon_lr_func(lr_init=0.0001,
                                                    lr_final=0.0001,
                                                    max_steps=opt['train_iterations'])
        self.position_scheduler = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.00001,
                                                    max_steps=opt['train_iterations'])
        self.scale_scheduler = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.00001,
                                                    max_steps=opt['train_iterations'])
        self.wave_scheduler = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.001,
                                                    #lr_delay_steps=total_iters//2,
                                                    max_steps=opt['train_iterations'])
        self.rotation_scheduler = get_expon_lr_func(lr_init=0.001,
                                                    lr_final=0.001,
                                                    max_steps=opt['train_iterations'])
        self.color_scheduler = get_expon_lr_func(lr_init=0.0001,
                                                    lr_final=0.0001,
                                                    max_steps=opt['train_iterations'])
        
        self.cumulative_gradients = torch.empty([0], device=device, dtype=torch.float32)

    def create_optimizer(self):
        l = [
            {'params': [self.primitive_alphas], 'lr': 0.005, "name": "primitive_alphas", "weight_decay": 0},
            {'params': [self.primitive_colors], 'lr': 0.005, "name": "primitive_colors", "weight_decay": 0},
            {'params': [self.primitive_positions], 'lr': 0.005, "name": "primitive_positions"},
            {'params': [self.primitive_scales], 'lr': 0.005, "name": "primitive_scales"},
            {'params': [self.primitive_rotations], 'lr': 0.005, "name": "primitive_rotations"},
            {'params': [self.wave_coefficients], 'lr': 0.005, "name": "wave_coefficients"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=[0.9, 0.999])
        return optimizer
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if("positions" in param_group['name']):
                param_group['lr'] = self.position_scheduler(iteration)
            if("alpha" in param_group['name']):
                param_group['lr'] = self.alpha_scheduler(iteration)
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
            if(group['name'] == 'wave_coefficients' and not self.opt['gaussian_only']):
                total += self.primitive_colors.shape[0]*(self.num_top_freqs + self.num_random_freqs)
            elif(group['name'] != 'wave_coefficients'):   
                total += group['params'][0].numel()
        return total
    
    def effective_param_count(self):
        return self.param_count()
    
    def get_weighed_frequency_dist(self):
        top_k_coeffs, top_k_indices = self.get_topk_waves(with_random=False)
        frequencies = top_k_indices*self.opt['max_frequency'] / self.opt['num_total_frequencies']
        return frequencies.flatten(), top_k_coeffs.flatten()

    def add_primitives_random(self, num_primitives):
        new_alphas = 0.8*torch.rand([num_primitives, 1], 
                dtype=torch.float32, device=self.opt['device'])
        new_colors = torch.rand([num_primitives, 3], 
                dtype=torch.float32, device=self.opt['device'])        
        new_positions = torch.rand([num_primitives, 3], 
                dtype=torch.float32, device=self.opt['device'])
        new_scales = -3 + torch.randn([num_primitives, 1], 
                dtype=torch.float32,  device=self.opt['device']).repeat(1, 3) + \
                0.05*torch.randn([num_primitives, 3], dtype=torch.float32,  device=self.opt['device'])
        #https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
        # h = ( sqrt(1-u) sin(2πv), sqrt(1-u) cos(2πv), sqrt(u) sin(2πw), sqrt(u) cos(2πw))
        u = torch.rand([num_primitives], device=self.opt['device'], dtype=torch.float32)
        v = torch.rand([num_primitives], device=self.opt['device'], dtype=torch.float32)
        w = torch.rand([num_primitives], device=self.opt['device'], dtype=torch.float32)
        new_rotations = torch.stack([
            ((1-u)**0.5)*torch.sin(2*torch.pi*v),
            ((1-u)**0.5)*torch.cos(2*torch.pi*v),
            (u**0.5)*torch.sin(2*torch.pi*w),
            (u**0.5)*torch.cos(2*torch.pi*w)
        ], dim = 1)
        if(not self.opt['gaussian_only']):
            new_wave_coefficients = 0.01*(2*torch.rand([num_primitives, self.opt['num_total_frequencies']],
                    dtype=torch.float32,  device=self.opt['device'])-1) 
            #new_wave_coefficients[:,0] = 1.0
        else:
            new_wave_coefficients = None

        tensor_dict = {
            "primitive_alphas": new_alphas,
            "primitive_colors": new_colors, 
            "primitive_positions": new_positions,
            "primitive_scales": new_scales,
            "primitive_rotations": new_rotations,
            "wave_coefficients": new_wave_coefficients
        }

        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.primitive_alphas = updated_params['primitive_alphas']
        self.primitive_colors = updated_params['primitive_colors']
        self.primitive_positions = updated_params['primitive_positions']
        self.primitive_scales = updated_params['primitive_scales']
        self.primitive_rotations = updated_params['primitive_rotations']
        if(not self.opt['gaussian_only']):
            self.wave_coefficients = updated_params['wave_coefficients']
        self.cumulative_gradients = torch.cat([self.cumulative_gradients, 
            torch.empty([num_primitives], device=self.opt['device'], dtype=torch.float32)])

    def get_num_primitives(self):
        return self.primitive_colors.shape[0]
    
    def add_primitives(self, num_primitives):
        self.add_primitives_random(num_primitives)
        return 0
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(group['name'] in tensors_dict):
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                if extension_tensor is None:
                    continue
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
    
    def split_prims(self, num_primitives):
        
        new_prims = min(num_primitives, self.get_num_primitives())
        _, indices = torch.topk(self.cumulative_gradients, new_prims)

        stds = 1/(2*torch.exp(self.primitive_scales[indices].clone()))
        #stds = stds*stds
        means = torch.zeros((stds.shape[0], 3), device=self.opt['device'], dtype=torch.float32)
        samples = torch.normal(mean=means, std=stds)
        new_rotations = self.primitive_rotations[indices].clone()
        #rotated_samples = torch.stack([samples[:,0]*torch.cos(-new_rotations[:,0]) + samples[:,1]*torch.sin(-new_rotations[:,0]),
        #                       samples[:,0]*-torch.sin(-new_rotations[:,0]) + samples[:,1]*torch.cos(-new_rotations[:,0])], dim=-1)

        new_positions = self.primitive_positions[indices].clone() + rotated_samples
        new_colors = self.primitive_colors[indices].clone() * 0.8
        new_scales = self.primitive_scales[indices].clone() + np.log(1.6)
        if(not self.opt['gaussian_only']):
            new_wave_coefficients = self.wave_coefficients[indices].clone()
            #new_wave_coefficients[:,1:] *= 0.01
            #new_wave_coefficients[:,0] = 1.
        else:
            new_wave_coefficients = None
        self.primitive_scales[indices] += np.log(1.1)
        self.primitive_colors[indices] *= 0.9

        tensor_dict = {
            "primitive_colors": new_colors, 
            "primitive_positions": new_positions,
            "primitive_scales": new_scales,
            "primitive_rotations": new_rotations,
            "wave_coefficients": new_wave_coefficients
        }
        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.primitive_colors = updated_params['primitive_colors']
        self.primitive_positions = updated_params['primitive_positions']
        self.primitive_scales = updated_params['primitive_scales']
        self.primitive_rotations = updated_params['primitive_rotations']
        if(not self.opt['gaussian_only']):
            self.wave_coefficients = updated_params['wave_coefficients']
        self.cumulative_gradients = torch.cat([self.cumulative_gradients, 
            torch.empty([new_prims], device=self.opt['device'], dtype=torch.float32)])
        self.cumulative_gradients.zero_()
        return new_prims
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        with torch.no_grad():
            dict = torch.load(path)
            for name in dict:
                if name == "primitive_alphas":
                    self.primitive_alphas = Parameter(dict[name])
                if name == "primitive_colors":
                    self.primitive_colors = Parameter(dict[name])
                if name == "primitive_positions":
                    self.primitive_positions = Parameter(dict[name])
                if name == "primitive_scales":
                    self.primitive_scales = Parameter(dict[name])
                if name == "primitive_rotations":
                    self.primitive_rotations = Parameter(dict[name])
                if name == "wave_coefficients":
                    self.wave_coefficients = Parameter(dict[name])

        print("Successfully loaded trained model.")
    
    def vis_heatmap(self, points):
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        heatmap = PeriodicPrimitivesFunction.apply(points, 
            self.primitive_colors, self.primitive_positions, torch.exp(self.primitive_scales),
            self.primitive_rotations, top_k_coeffs, top_k_indices,
            self.num_top_freqs, self.num_random_freqs, self.opt['max_frequency'], self.opt['gaussian_only'], True)
        heatmap = heatmap / heatmap.max()
        return heatmap

    def prune_tensors_from_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(self.opt['gaussian_only'] and group['name'] == "wave_coefficients"):
                continue
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

    def prune_primitives(self, min_contribution:int=1./1000., min_width = 8192.):
        
        to_remove = 0
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
        if(self.training or True):
            coefficients_to_send, indices_to_send = torch.topk(torch.abs(self.wave_coefficients),
                                    self.opt['num_total_frequencies'], dim=1, sorted=False)
            coefficients_to_send = coefficients_to_send * \
                torch.gather(self.wave_coefficients.sign(), dim=1, index=indices_to_send)
            indices_to_send = indices_to_send.type(torch.int)
        
        return coefficients_to_send, indices_to_send

    def forward(self, cam_position, view_matrix, VP_matrix, 
                fov_x, fov_y, 
                image_width, image_height,
                background_color,
                scale_modifier = 1.0,
                gaussian_only = None) -> torch.Tensor:
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        if(gaussian_only is None): 
            gaussian_only = self.opt['gaussian_only']
        return PeriodicPrimitivesFunction.apply(self.primitive_colors, 
                self.primitive_alphas, background_color,
                self.primitive_positions, 
                torch.exp(self.primitive_scales), 
                scale_modifier,
                self.primitive_rotations / (torch.norm(self.primitive_rotations)+1e-8), 
                top_k_coeffs, top_k_indices,
                self.opt['max_frequency'], 
                cam_position, view_matrix, VP_matrix,
                fov_x, fov_y, 
                image_width, image_height,
                gaussian_only, False)

torch.random.manual_seed(42)
np.random.seed(42)
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.camera_utils import getProjectionMatrix
from options import Options
opt = Options.get_default()
opt['gaussian_only'] = True
model = PeriodicPrimitives3D(opt)
model.add_primitives(100)

fov_x = 60
fov_y = 60

background_color = torch.tensor([0.0, 0.0, 0.0], device="cuda")
cam_position = torch.tensor([0.0, 0.0, 0.0], device="cuda") # doesn't matter
view_matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, -10.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]], device="cuda")
proj_matrix = getProjectionMatrix(0.1, 10000, fov_x, fov_y).cuda()
VP_matrix = view_matrix @ proj_matrix
image_width = 800
image_height = 800

from time import time
t0 = time()
frames = 200
d = []
for i in range(frames):
    out = model.forward(cam_position, view_matrix, VP_matrix, 
                fov_x, fov_y, image_width, image_height, background_color)
    d.append((out.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
    view_matrix[1,-1] += 0.1
    VP_matrix = view_matrix @ proj_matrix
    #print(f"{i}: {out[0,0,0].item()}")
t1 = time()
print(f"{frames/(t1 - t0)} FPS")
import matplotlib.pyplot as plt
print(out.min())
print(out.max())
#plt.imshow(out.permute(1, 2, 0).detach().cpu().numpy())
#plt.show()
import imageio.v3 as imageio
imageio.imwrite("test.gif", d)