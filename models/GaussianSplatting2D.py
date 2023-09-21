import torch
from torch.nn.parameter import Parameter
from utils.math_utils import inv2x2
from utils.data_utils import to_img, psnr
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
torch.backends.cuda.matmul.allow_tf32 = True

class GaussianSplatting2D(torch.nn.Module):

    def __init__(self, n_primitives, n_channels = 1, device="cuda"):
        super().__init__()
        self.device=device
        self.n_primitives = n_primitives
        self.n_channels = n_channels
        self.colors = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.means = Parameter(torch.empty(0, device=device, dtype=torch.float32))        
        self.rotations = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        self.scales = Parameter(torch.empty(0, device=device, dtype=torch.float32))
        #self.offset = Parameter(torch.zeros([1], device=device, dtype=torch.float32))

        self.optimizer = self.create_optimizer()

    def create_optimizer(self):
        l = [
            {'params': [self.means], 'lr': 0.005, "name": "means"},
            {'params': [self.colors], 'lr': 0.001, "name": "colors"},
            {'params': [self.rotations], 'lr': 0.005, "name": "rotations"},
            {'params': [self.scales], 'lr': 0.005, "name": "scales"},
        ]

        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        return optimizer
    
    def param_count(self):
        total = 0
        for group in self.optimizer.param_groups:           
            total += group['params'][0].numel()
        return total
    
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

    def add_primitives(self, num_to_add = 10):

        new_colors = (0.05) - \
            0.1*(torch.rand([num_to_add, self.n_channels], dtype=torch.float32, device=self.device))
        new_means = torch.rand([num_to_add, 2], dtype=torch.float32, device=self.device)
        new_rotations = torch.pi*torch.rand([num_to_add, 1], dtype=torch.float32, device=self.device)
        new_scales = 3.-0.5*torch.rand([num_to_add, 1], dtype=torch.float32, device=self.device).expand(-1, 2)

        tensor_dict = {
            "colors": new_colors, 
            "means": new_means,
            "rotations": new_rotations,
            "scales": new_scales
        }

        updated_params = self.cat_tensors_to_optimizer(tensor_dict)

        self.colors = updated_params['colors']
        self.means = updated_params['means']
        self.scales = updated_params['scales']
        self.rotations = updated_params['rotations']
        return num_to_add

    def prune_primitives(self, min_contribution=1./255.):
        if(self.means.shape[0] == 0):
            return
        mask = torch.linalg.norm(self.colors,dim=-1) > min_contribution
        if(len(mask.shape)>0):
            to_remove = mask.shape[0]-mask.sum()
            if(to_remove>0):
                #print(f" Pruning {to_remove} wave{'s' if to_remove>1 else ''}.")
                updated_params = self.prune_tensors_from_optimizer(mask)

                self.colors = updated_params['colors']
                self.means = updated_params['means']
                self.scales = updated_params['scales']
                self.rotations = updated_params['rotations']

    def vis_primitives(self, x, res=[256, 256], power=10):

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
            cov = RS @ RS.mT
            #cov = inv2x2(cov)
            # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
            #transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
            transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
            # [N, n_gaussians, 1, 1]
            gauss_vals = self.colors[None,...]*torch.exp(-(transformed_x[:,:,0])/2)
            
            vals = gauss_vals

            im_per_row = 4
            nwaves = max(self.means.shape[0], self.n_primitives)
            n_cols = min(nwaves, im_per_row)
            n_rows = 1+(nwaves//im_per_row)
            im = torch.zeros([n_rows*res[0], n_cols*res[1], self.n_channels])
            
            row_spot = 0
            col_spot = 0
            
            for i in range(vals.shape[1]):

                im[row_spot*res[0]:(row_spot+1)*res[0],
                   col_spot*res[1]:(col_spot+1)*res[1]] = vals[:,i].detach().cpu().reshape(res+[self.n_channels])

                if((i+1) % im_per_row == 0):
                    row_spot += 1
                    col_spot = 0
                else:
                    col_spot += 1
        return im
    
    def train_model(self, x, y, im_shape):
        #self.offset = Parameter(y.mean(dim=0))
        self.n_channels = y.shape[1]
        num_chunks = 10
        total_iterations = 10000

        gaussians_per_chunk = self.n_gaussians // num_chunks
        add_chunks_every = total_iterations // num_chunks
        
        p = 0

        num_params = []
        psnrs = []
        pre_fitting_imgs = []

        max_ls_points = 2**17
        pct_of_data = max_ls_points / x.shape[0]

        training_imgs = []
        gauss_imgs = []
        t = tqdm(range(total_iterations))
        for i in t:
            mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data
            # add new gaussians
            if i % add_chunks_every == 0:
                with torch.no_grad():          
                    self.prune_gaussians(1./500.)     
                    _ = self.add_more_gaussians(gaussians_per_chunk) 

            # image logging
            if i % 50 == 0 and i > 0:
                with torch.no_grad():
                    res = [256, 256]
                    xmin = x.min(dim=0).values
                    xmax = x.max(dim=0).values
                    g = [torch.linspace(xmin[i], xmax[i], res[i], device=self.device) for i in range(xmin.shape[0])]
                    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                    img = self(g).reshape(res + [self.n_channels])
                    img = to_img(img)
                    training_imgs.append(img)
                    gauss_imgs.append(self.vis_each_gaussian(g))

            self.optimizer.zero_grad()
            mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data
            loss, model_out = self.loss(x[mask], y[mask])
            loss.backward()
            self.optimizer.step()
            t.set_description(f"[{i+1}/{total_iterations}] loss: {loss.item():0.04f}")

        self.prune_gaussians(1./500.) 
        
        imageio.imwrite("output/gauss_training.mp4", training_imgs)
        imageio.imwrite("output/each_gauss_training.mp4", gauss_imgs)
        
        with torch.no_grad():
            model_out = self(x)
            p = psnr(model_out, y).item()
            img = to_img(model_out.reshape(im_shape))
            err = to_img((torch.abs(y - model_out)).reshape(im_shape))
            err_rec = np.concatenate([
                img,
                err,
            ], axis=1)
            GT = np.concatenate([
                to_img(y.reshape(im_shape)),
                to_img(y.reshape(im_shape))
                ], axis=1)
            pre_fitting_imgs.append(np.concatenate([err_rec, GT], axis=0))
            imageio.imwrite("output/gaussian_prefitting.mp4", pre_fitting_imgs)
            print(f"Number of added gaussians: {self.means.shape[0]}")
            p = psnr(self(x),y).item()
            print(f"Final PSNR: {p:0.02f}")
        
        psnrs.append(p)
        num_params.append(self.means.shape[0]*6+1)

        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.plot(num_params, psnrs, color="blue")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_xlabel("Num params")        
        ax1.set_title("Reconstruction vs # params")
        plt.savefig("output/training_details.png")

    def create_RS(self):
        return torch.stack([torch.exp(self.scales[:,0:1])*torch.cos(self.rotations), 
                            -torch.exp(self.scales[:,1:2])*torch.sin(self.rotations),
                            torch.exp(self.scales[:,0:1])*torch.sin(self.rotations), 
                            torch.exp(self.scales[:,1:2])*torch.cos(self.rotations)], 
                            dim=-1).reshape(-1, 2, 2)

    def loss(self, x, y):
        # x is our output, y is the ground truth
        model_out = self(x)
        mse = torch.nn.functional.mse_loss(model_out,y)
        centering_loss = 0.0001*torch.abs(self.means - 0.5).mean()
        decay_loss = 0.001*torch.abs(self.colors).mean()
        final_loss = mse #+ centering_loss
        losses = {
            "final_loss": final_loss,
            "mse": mse
        }
        return losses, model_out
    
    def forward(self, x):
        # x is [N, 2]
        rel_x = x[:,None,:] - self.means[:,...] # [N, n_gaussians, 2]
        RS = self.create_RS() # [n_gaussians, 2, 2]
        cov = RS @ RS.mT
        #cov = inv2x2(cov)
        # [N, n_gaussians, 2, 1] x [1, n_gaussians, 2, 2] x [N, n_gaussians, 2, 1]
        transformed_x = rel_x[...,None].mT @ cov[None,...] @ rel_x[...,None]
        # [N, n_gaussians, 1, 1]
        transformed_x = torch.exp(-transformed_x[:,:,0]/2) # [N, n_gaussians]
        vals = self.colors[None,...]*transformed_x # [N, n_gaussians, channels]
        vals = torch.sum(vals, dim=1)#+self.offset
        return vals # [N, channels]

