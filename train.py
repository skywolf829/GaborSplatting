
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KINETO_LOG_LEVEL'] = '3'

def log_scalars(writer, model, losses, i):
    with torch.no_grad():
        writer.add_scalar("Loss", losses['final_loss'].item(), i)     
        p = 20*np.log10(1.0) - 10*torch.log10(losses['mse'])
        writer.add_scalar("Train PSNR", p, i)
        writer.add_scalar("Num primitives", model.get_num_primitives(), i)    
        writer.add_scalar("Param count", model.param_count(), i)
        writer.add_scalar("Effective param count", model.effective_param_count(), i) 
        writer.add_scalar("Params vs. PSNR", 
                            p, model.param_count())
        writer.add_scalar("Effective params vs. PSNR", 
                            p, model.param_count())
    return p.item()

def log_imgs(writer, model, g, img_shape, i):
    with torch.no_grad():     
        img = model(g).reshape(img_shape+[model.n_channels])
        img = to_img(img)
        writer.add_image('reconstruction', img, i, dataformats='HWC')
        heatmap = model.vis_heatmap(g).reshape(img_shape+[model.n_channels])
        heatmap = to_img(heatmap)
        writer.add_image('heatmap', heatmap, i, dataformats='HWC')
       
def log_frequencies(writer, model, i):
    if(not model.gaussian_only):
        freqs = model.get_weighed_frequency_dist()
        writer.add_histogram("Frequency distribution", freqs, i)
    writer.add_histogram("Scale distribution", model.gaussian_scales.flatten(), i)

if __name__ == '__main__':
    
    total_iters = 30000
    fine_tune_iters = 5000  
    starting_primitives = 2560
    total_primitives = 50000
    only_gaussians = True
    
    split_every = 1000
    prune_every = 100
    black_out_every = 3000

    img_name = "pluto.png"

    device = "cuda"
    torch.random.manual_seed(42)
    np.random.seed(42)
    training_img = load_img("./data/"+img_name)
    img_shape = list(training_img.shape)[0:2]
    training_img_copy = (training_img.copy() * 255).astype(np.uint8)
    og_img_shape = list(training_img.shape)
    model = PeriodicPrimitives2D(device=device, n_channels=3, gaussian_only=only_gaussians, total_iters=total_iters)
    n_extracted_peaks = model.add_primitives(starting_primitives)

    g_x = torch.arange(0, og_img_shape[0], dtype=torch.float32, device=device) / (og_img_shape[0]-1)
    g_y = torch.arange(0, og_img_shape[1], dtype=torch.float32, device=device) / (og_img_shape[1]-1)
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)
    training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=device).reshape(-1,training_img.shape[-1])

    x = training_img_positions
    y = training_img_colors 
    
    max_img_reconstruction_dim_size = 1024    
    batch_size = 2**17
    pct_of_data = batch_size / x.shape[0]
    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values

    img_scale = max_img_reconstruction_dim_size/max(img_shape)
    scaled_img_shape = [int(img_shape[i]*img_scale) for i in range(xmin.shape[0])]
    g = [torch.linspace(xmin[i], xmax[i], scaled_img_shape[i], device=model.device) for i in range(xmin.shape[0])]
    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
    current_t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = model.__class__.__name__+"_"+img_name.split('.')[0]+"_"+current_t
    writer = SummaryWriter(f"./runs/{run_name}")
    t = tqdm(range(total_iters))
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=10,
            active=10,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./runs/traces/{run_name}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,

    ) as prof:
        for i in t:
            
            # image logging
            if i % 1000 == 0 and i > 0:
                log_imgs(writer, model, g, scaled_img_shape, i)
            # histogram logging
            if i % 1000 == 0 and i > 0:
                log_frequencies(writer, model, i)

            model.update_learning_rate(i)
            mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

            # Prune primitives
            if i % prune_every == 0 and i > 0 and prune_every > 0:
                with torch.no_grad():
                    prims_removed = model.prune_primitives(1./500.)
                    writer.add_scalar("Primitives pruned", prims_removed, i)

            # split primitives
            if i % split_every == 0 and i < total_iters-fine_tune_iters and split_every != -1 and model.get_num_primitives() < total_primitives:          
                num_to_go = total_primitives - model.get_num_primitives()
                iters_to_go = total_iters-fine_tune_iters - i
                splits_left = int(iters_to_go/split_every)
                num_to_add = int(num_to_go/splits_left)
                if(num_to_add > 0):
                    if(model.get_num_primitives() > 0):
                        with torch.no_grad():
                            new_prims = model.split_prims(num_to_add)
                            writer.add_scalar("Prims split", new_prims, i)
                        model.zero_grad()
                    else:
                        with torch.no_grad():
                            n_extracted_peaks = model.add_primitives(starting_primitives)
            if i % black_out_every == 0 and i > 0 and black_out_every > 0 and i < total_iters - fine_tune_iters:
                with torch.no_grad():
                    model.gaussian_colors *= 0.001
            
            model.optimizer.zero_grad()
            losses, model_out = model.loss(x[mask], y[mask])
            losses['final_loss'].backward()
            with torch.no_grad():
                model.cumulative_gradients *= 0.95
                model.cumulative_gradients += torch.norm(model.gaussian_positions.grad, dim=1)
                model.cumulative_gradients += torch.norm(model.gaussian_scales.grad, dim=1)
            model.optimizer.step()
                
            # logging
            if i % 100 == 0:
                p = log_scalars(writer, model, losses, i)
                t.set_description(f"[{i+1}/{total_iters}] PSNR: {p:0.04f}")
            prof.step()

    with torch.no_grad():
        output = torch.empty([x.shape[0], model.n_channels], dtype=torch.float32, device=model.device)
        max_batch = int((2**28) / 256.)
        for i in range(0, x.shape[0], max_batch):
            end = min(output.shape[0], i+max_batch)
            output[i:end] = model(x[i:end])
        final_im = to_img(output.reshape(og_img_shape))
        print(final_im.shape)
        imageio.imwrite(os.path.join("./output", run_name+".jpg"), final_im)
        #writer.add_image("Final image", final_im)
        p = psnr(output,y).item()
        print(f"Final PSNR: {p:0.02f}")
        writer.add_scalar("Params vs. PSNR", 
                        p, model.param_count())
        writer.add_scalar("Effective params vs. PSNR", 
                        p, model.effective_param_count())
    writer.flush()
    writer.close()
    model.save(os.path.join("./savedModels", run_name+".ckpt"))
    
