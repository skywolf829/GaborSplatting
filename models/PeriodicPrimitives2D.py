import torch
from torch.utils.cpp_extension import load
from torch.nn import Parameter
import os
import numpy as np
import PeriodicPrimitives as periodic_primitives

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

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
    def forward(ctx, x, gaussian_colors, 
                gaussian_positions, gaussian_scales,
                gaussian_rotations, topk_wave_coefficients,
                topk_wave_indices,
                num_top_frequencies, num_random_frequencies,
                max_frequency, gaussian_only, heatmap=False):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        outputs = periodic_primitives.forward(x, 
            gaussian_colors, gaussian_positions, gaussian_scales,
            gaussian_rotations, topk_wave_coefficients, topk_wave_indices, 
            max_frequency, gaussian_only, heatmap)
        
        result, sorted_gaussian_indices_tensor, \
        blocks_gaussian_start_end_indices_tensor, \
        sorted_query_point_indices, blocks_query_points_start_end_indices_tensor = outputs
        
        variables = [x, gaussian_colors, gaussian_positions, 
                    gaussian_scales, gaussian_rotations,
                    topk_wave_coefficients, topk_wave_indices, 
                    sorted_gaussian_indices_tensor, 
                    blocks_gaussian_start_end_indices_tensor, 
                    sorted_query_point_indices, 
                    blocks_query_points_start_end_indices_tensor]
        
        ctx.save_for_backward(*variables)
        ctx.max_frequency = max_frequency
        ctx.gaussian_only = gaussian_only
        return result.T

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
        #for t in ctx.saved_tensors:
        #    del t
        return grad_output, grad_gaussian_colors, grad_gaussian_positions, \
              grad_gaussian_scales, grad_gaussian_rotations, grad_wave_coefficients, \
              None, None, None, None, None, None


class PeriodicPrimitives2D(torch.nn.Module):
    def __init__(self, opt):
        super(PeriodicPrimitives2D, self).__init__()
        self.loaded = False
        self.wave_coefficients = None
        device = opt['device']
        self.opt = opt

        self.gaussian_colors = Parameter(torch.empty([0, opt['num_outputs']], device=device, dtype=torch.float32)) 
        self.gaussian_positions = Parameter(torch.empty([0, opt['num_dims']], device=device, dtype=torch.float32))  
        self.gaussian_scales = Parameter(torch.empty([0, opt['num_dims']], device=device, dtype=torch.float32))    
        self.gaussian_rotations = Parameter(torch.empty([0, 1], device=device, dtype=torch.float32))      
        self.wave_coefficients = Parameter(torch.empty([0, opt['num_total_frequencies']], device=device, dtype=torch.float32))
        
        self.optimizer = self.create_optimizer()
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
            {'params': [self.gaussian_colors], 'lr': 0.005, "name": "gaussian_colors", "weight_decay": 0},
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
            if(group['name'] == 'wave_coefficients' and not self.opt['gaussian_only']):
                total += self.gaussian_colors.shape[0]*(self.opt['num_frequencies'])
            elif(group['name'] != 'wave_coefficients'):   
                total += group['params'][0].numel()
        return total
    
    def effective_param_count(self):
        total = 0
        mult = 1.0
        for group in self.optimizer.param_groups:    
            if(group['name'] == 'wave_coefficients' and not self.opt['gaussian_only']):
                top_k_coeffs, _ = self.get_topk_waves()
                above_thresh = torch.abs(top_k_coeffs) > 1./1000.
                total += above_thresh.sum()
            if(group['name'] == "gaussian_colors"):
                mult = self.gaussian_colors.norm(dim=-1) > 1/1000.
                mult = mult.sum() / self.gaussian_colors.shape[0]

            if(group['name'] != 'wave_coefficients'):   
                total += group['params'][0].numel()
        return total * mult
    
    def get_weighed_frequency_dist(self):
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        frequencies = top_k_indices*self.opt['max_frequency'] / self.opt['num_total_frequencies']
        return frequencies.flatten(), top_k_coeffs.flatten()

    def add_primitives_random(self, num_gaussians):
        new_colors = 0.05*(torch.randn([num_gaussians, self.opt['num_outputs']], 
                dtype=torch.float32, device=self.opt['device']))
        new_positions = torch.rand([num_gaussians, self.opt['num_dims']], 
                dtype=torch.float32, device=self.opt['device'])
        new_scales = 3 + torch.randn([num_gaussians, 1], 
                dtype=torch.float32,  device=self.opt['device']).repeat(1, 2) + \
                0.05*torch.randn([num_gaussians, 2], dtype=torch.float32,  device=self.opt['device'])
        new_rotations = torch.pi*torch.rand([num_gaussians, 1],
                dtype=torch.float32,  device=self.opt['device'])
        if(not self.opt['gaussian_only']):
            new_wave_coefficients = 0.01*(2*torch.rand([num_gaussians, self.opt['num_total_frequencies']],
                    dtype=torch.float32,  device=self.opt['device'])-1) 
            new_wave_coefficients[:,0] = 0.01
        else:
            new_wave_coefficients = None

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
        if(not self.opt['gaussian_only']):
            self.wave_coefficients = updated_params['wave_coefficients']
        self.cumulative_gradients = torch.cat([self.cumulative_gradients, 
            torch.empty([num_gaussians], device=self.opt['device'], dtype=torch.float32)])

    def get_num_primitives(self):
        return self.gaussian_colors.shape[0]
    
    def add_primitives(self, num_primitives):
        self.add_primitives_random(num_primitives)
        return 0
    
    def training_routine_updates(self, i, writer=None):
        self.update_learning_rate(i)
        
        # Prune primitives
        if i % self.opt['prune_every_iters'] == 0 and i > 0 and self.opt['prune_every_iters'] > 0:
            with torch.no_grad():
                prims_removed = self.prune_primitives(min_contribution=1./1000.)
                
                if(writer is not None):
                    writer.add_scalar("Primitives pruned", prims_removed, i)
            torch.cuda.empty_cache()
            
        # split primitives
        if i % self.opt['split_every_iters'] == 0 and i < self.opt['train_iterations']-self.opt['fine_tune_iterations'] and \
            self.opt['split_every_iters'] != -1 and self.get_num_primitives() < self.opt['num_total_prims']:          
            num_to_go = self.opt['num_total_prims'] - self.get_num_primitives()
            iters_to_go = self.opt['train_iterations']-self.opt['fine_tune_iterations'] - i
            splits_left = max(int(iters_to_go/self.opt['split_every_iters']), 1)
            num_to_add = int(num_to_go/splits_left)
            if(num_to_add > 0):
                if(self.get_num_primitives() > 0):
                    with torch.no_grad():
                        new_prims = self.split_prims(num_to_add)
                        if(writer is not None):
                            writer.add_scalar("Prims split", new_prims, i)
                    self.zero_grad()
                else:
                    with torch.no_grad():
                        self.add_primitives(self.opt['num_starting_prims'])
            torch.cuda.empty_cache()

        if i % self.opt['blackout_every_iters'] == 0 and i > 0 and self.opt['blackout_every_iters'] > 0 and \
            i < self.opt['train_iterations'] - self.opt['fine_tune_iterations']:
            with torch.no_grad():
                self.gaussian_colors *= 0.001
            torch.cuda.empty_cache()

    def update_cumulative_gradients(self):
        with torch.no_grad():
            self.cumulative_gradients *= 0.95
            self.cumulative_gradients += torch.norm(self.gaussian_positions.grad, dim=1)
            self.cumulative_gradients += torch.norm(self.gaussian_scales.grad, dim=1)

    def replace_optimizer_tensors(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if(group['name'] in tensors_dict):
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                if extension_tensor is None:
                    continue
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.zeros_like(extension_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(extension_tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = Parameter(extension_tensor).requires_grad_(True)
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = Parameter(extension_tensor).requires_grad_(True)
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

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

        stds = 1/(2*torch.exp(self.gaussian_scales[indices].clone()))
        #stds = stds*stds
        means = torch.zeros((stds.shape[0], 2), device=self.opt['device'], dtype=torch.float32)
        samples = torch.normal(mean=means, std=stds)
        new_rotations = self.gaussian_rotations[indices].clone()
        rotated_samples = torch.stack([samples[:,0]*torch.cos(-new_rotations[:,0]) + samples[:,1]*torch.sin(-new_rotations[:,0]),
                               samples[:,0]*-torch.sin(-new_rotations[:,0]) + samples[:,1]*torch.cos(-new_rotations[:,0])], dim=-1)
        new_positions = self.gaussian_positions[indices].clone() + rotated_samples
        new_colors = self.gaussian_colors[indices].clone() * 0.8
        new_scales = self.gaussian_scales[indices].clone() + np.log(1.6)
        if(not self.opt['gaussian_only']):
            new_wave_coefficients = self.wave_coefficients[indices].clone()
            #new_wave_coefficients[:,1:] *= 0.01
            #new_wave_coefficients[:,0] = 1.
        else:
            new_wave_coefficients = None
        self.gaussian_scales[indices] += np.log(1.1)
        self.gaussian_colors[indices] *= 0.9

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
        if(not self.opt['gaussian_only']):
            self.wave_coefficients = updated_params['wave_coefficients']
        self.cumulative_gradients = torch.cat([self.cumulative_gradients, 
            torch.empty([new_prims], device=self.opt['device'], dtype=torch.float32)])
        self.cumulative_gradients.zero_()
        return new_prims
        
    def save(self, path):
        p = self.gaussian_positions.detach().cpu().numpy().astype(np.float32)
        s = self.gaussian_scales.detach().cpu().numpy().astype(np.float32)
        c = self.gaussian_colors.detach().cpu().numpy().astype(np.float32)
        r = self.gaussian_rotations.detach().cpu().numpy().astype(np.float32)
        f, i = self.get_topk_waves()
        f = f.detach().cpu().numpy().astype(np.float32)
        i = i.detach().cpu().numpy().astype(np.uint8)
        np.savez_compressed(os.path.join(path, "model.ckpt"),
                        positions=p,
                        scales=s,
                        colors=c,
                        rotations=r,
                        frequency_coefficients=f,
                        frequency_indices=i)
        #torch.save(self.state_dict(), os.path.join(path, "model.ckpt"))

    def load(self, path):
        with torch.no_grad():
            
            data = np.load(os.path.join(path, "model.ckpt.npz"))
            c = torch.tensor(data['colors'], device=self.opt['device'])
            p = torch.tensor(data['positions'], device=self.opt['device'])
            s = torch.tensor(data['scales'], device=self.opt['device'])
            r = torch.tensor(data['rotations'], device=self.opt['device'])
            if 'frequency_coefficients' in data.keys():
                f = torch.tensor(data['frequency_coefficients'], device=self.opt['device'])
                i = torch.tensor(data['frequency_indices'].astype(np.int32), device=self.opt['device'])
            else:
                f = None

            tensor_dict = {
                "gaussian_colors": c, 
                "gaussian_positions": p,
                "gaussian_scales": s,
                "gaussian_rotations": r,
                "wave_coefficients": f
            }
            updated_params = self.replace_optimizer_tensors(tensor_dict)

            self.gaussian_colors = updated_params['gaussian_colors']
            self.gaussian_positions = updated_params['gaussian_positions']
            self.gaussian_scales = updated_params['gaussian_scales']
            self.gaussian_rotations = updated_params['gaussian_rotations']
            
            if(not self.opt['gaussian_only']):
                self.wave_coefficients = updated_params['wave_coefficients']
                self.wave_coefficient_indices = i
            self.cumulative_gradients = torch.zeros([c.shape[0]], device=self.opt['device'], dtype=torch.float32)

        print("Successfully loaded trained model.")
        self.loaded = True
    
    def vis_heatmap(self, points):
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        heatmap = PeriodicPrimitivesFunction.apply(points, 
            self.gaussian_colors, self.gaussian_positions, torch.exp(self.gaussian_scales),
            self.gaussian_rotations, top_k_coeffs, top_k_indices,
            self.opt['num_frequencies'] , 0, self.opt['max_frequency'], self.opt['gaussian_only'], True)
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

    def prune_primitives(self, min_contribution:int=1./1000., min_width = 20000.):
        self.gaussian_scales.clamp_max_(np.log(min_width*0.99))
        exp_scales = torch.exp(self.gaussian_scales)
        r_max = 3/exp_scales.min(dim=1).values
        inv_r_min = exp_scales.max(dim=1).values
        on_image = (self.gaussian_positions[:,0]+r_max >= 0) * (self.gaussian_positions[:,0]-r_max <= 1.0) * \
                    (self.gaussian_positions[:,1]+r_max >= 0) * (self.gaussian_positions[:,1]-r_max <= 1.0)
        large_enough = inv_r_min < min_width
        gaussians_mask = on_image * large_enough
        
        if(not self.opt['gaussian_only']):
            #gaussians_mask *= (torch.linalg.norm(self.gaussian_colors,dim=-1) > min_contribution)
            gaussians_mask *= torch.linalg.norm(self.wave_coefficients, dim=-1) > min_contribution
        #else:
        #    gaussians_mask *= (torch.linalg.norm(self.gaussian_colors,dim=-1) > min_contribution)
        
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
                if not self.opt['gaussian_only']:
                    self.wave_coefficients = updated_params['wave_coefficients']
                self.cumulative_gradients = self.cumulative_gradients[gaussians_mask]
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
        if(not self.loaded):
            coefficients_to_send, indices_to_send = torch.topk(torch.abs(self.wave_coefficients),
                                    self.opt['num_frequencies'], dim=1, sorted=False)
            coefficients_to_send = coefficients_to_send * torch.gather(self.wave_coefficients.sign(), dim=1, index=indices_to_send)
            indices_to_send = indices_to_send.type(torch.int)
            #coefficients_to_send, indices_to_send = torch.max(
            #    torch.abs(self.wave_coefficients), dim=1, keepdim=True)
            #coefficients_to_send = coefficients_to_send * torch.gather(self.wave_coefficients.sign(), dim=1, index=indices_to_send)
            #indices_to_send = indices_to_send.type(torch.int)
        else:
            # not implemented yet
            coefficients_to_send = self.wave_coefficients if not self.opt['gaussian_only'] else torch.empty([0, 0], device=self.opt['device'])
            indices_to_send = self.wave_coefficient_indices if not self.opt['gaussian_only'] else torch.empty([0, 0], dtype=torch.int32, device=self.opt['device'])
        
        return coefficients_to_send, indices_to_send

    def forward(self, x) -> torch.Tensor:
        top_k_coeffs, top_k_indices = self.get_topk_waves()
        return PeriodicPrimitivesFunction.apply(x, 
            self.gaussian_colors, self.gaussian_positions, torch.exp(self.gaussian_scales),
            self.gaussian_rotations, top_k_coeffs, top_k_indices,
            self.opt['num_frequencies'], 0, self.opt['max_frequency'] ,self.opt['gaussian_only'], False)

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
            w = top_k_coeffs[None,...]*torch.cos(rs_x[...,None]*self.max_frequency*(top_k_indices[None,...])/self.num_frequencies)
            w = w.sum(dim=2)
            g = g*w
        return g@self.gaussian_colors 