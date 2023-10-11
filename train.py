
print("Loading HybridPrimitives CUDA kernel. May need to compile...")
#from models.HybridPrimitives import HybridPrimitives
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
print("Successfully loaded HybridPrimitives.")
#from models.Siren import Siren
from utils.data_generators import load_img
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm
from utils.data_utils import to_img, psnr
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def log_scalars(writer, model, losses, i):
    with torch.no_grad():
        writer.add_scalar("Loss", losses['final_loss'].item(), i)     
        p = 20*np.log10(1.0) - 10*torch.log10(losses['mse'])
        writer.add_scalar("Train PSNR", p, i)
        writer.add_scalar("Num primitives", model.get_num_primitives(), i)     
        writer.add_scalar("Params vs. PSNR", 
                            20*np.log10(1.0) - 10*torch.log10(losses['mse']), 
                            model.param_count())
    return p.item()

def log_imgs(writer, model, g, i):
    with torch.no_grad():     
        img = model(g).reshape(img_shape+[model.n_channels])
        img = to_img(img)
        writer.add_image('reconstruction', img, i, dataformats='HWC')
        heatmap = model.vis_heatmap(g).reshape(img_shape)
        heatmap = to_img(heatmap)[...,None]
        writer.add_image('heatmap', heatmap, i, dataformats='HWC')
       
if __name__ == '__main__':
    
    total_iters = 10000
    fine_tune_iters = 1000  
    starting_primitives = 100
    total_primitives = 1000
    primitives_per_update = 50
    iters_per_primitive = int((total_iters-fine_tune_iters) / (total_primitives/primitives_per_update))
    start_freq = 20
    end_freq = 512


    prune_every = 100
    split_every = 250

    model_type = PeriodicPrimitives2D
    img_name = "truck.jpg"

    device = "cuda"
    torch.random.manual_seed(42)
    np.random.seed(42)
    training_img = load_img("./data/"+img_name)
    img_shape = list(training_img.shape)[0:2]
    training_img_copy = (training_img.copy() * 255).astype(np.uint8)
    og_img_shape = training_img.shape
    model = model_type(device=device, n_channels=3, gaussian_only=True)
    n_extracted_peaks = model.add_primitives(starting_primitives)

    g_x = torch.arange(0, og_img_shape[0], dtype=torch.float32, device=device) / (og_img_shape[0]-1)
    g_y = torch.arange(0, og_img_shape[1], dtype=torch.float32, device=device) / (og_img_shape[1]-1)
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)
    training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=device).reshape(-1,training_img.shape[-1])

    x = training_img_positions
    y = training_img_colors 
    

    batch_size = 2**17
    pct_of_data = batch_size / x.shape[0]
    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values
    g = [torch.linspace(xmin[i], xmax[i], img_shape[i], device=model.device) for i in range(xmin.shape[0])]
    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
    current_t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model.__class__.__name__+"_"+img_name.split('.')[0]+current_t
    writer = SummaryWriter(f"./runs/{img_name.split('.')[0]}/{model.__class__.__name__}/{current_t}")
    t = tqdm(range(total_iters))
    for i in t:
        model.update_learning_rate(i)
        mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

        # Prune primitives
        if i % prune_every == 0 and i > 0:
            with torch.no_grad():
                model.prune_primitives(1./500.)

        # split primitives
        if i % split_every == 0 and i < total_iters-fine_tune_iters and split_every != -1:                  
            if(model.get_num_primitives() > 0):
                log_imgs(writer, model, g, i)
                losses, model_out = model.loss(x[mask], y[mask])
                losses['final_loss'].backward()
                p = model.gaussian_positions.grad.detach().clone()
                s = model.gaussian_rotations.grad.detach().clone()
                r = model.gaussian_rotations.grad.detach().clone()
                with torch.no_grad():
                    model.split_prims(p, s, r, primitives_per_update)
                model.zero_grad()
                log_imgs(writer, model, g, i+1)
            else:
                with torch.no_grad():
                    n_extracted_peaks = model.add_primitives(starting_primitives)

        
        model.optimizer.zero_grad()
        losses, model_out = model.loss(x[mask], y[mask])
        losses['final_loss'].backward()
        model.optimizer.step()
            
        # logging
        if i % 20 == 0:
            p = log_scalars(writer, model, losses, i)
            t.set_description(f"[{i+1}/{total_iters}] PSNR: {p:0.04f}")
            
        # image logging
        #if i % 20 == 0 and i > 0:
            #log_imgs(writer, model, g, i)
                

    with torch.no_grad():
        spot = 0
        output = model(x)
        p = psnr(output,y).item()
        print(f"Final PSNR: {p:0.02f}")
        err = torch.clamp(((y-output)**2), 0., 1.)**0.5
        writer.add_scalar("Params vs. PSNR", 
                        p, 
                        model.param_count())
    writer.flush()
    writer.close()
    model.save(os.path.join("./savedModels", run_name+".ckpt"))
    
