
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
from torch.profiler import profile, record_function, ProfilerActivity

from datasets.datasets import create_dataset
from models.models import create_model, load_model
from models.options import Options, load_options, save_options, update_options_from_args
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

project_folder_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def zoom_test(model, res, start_rect, end_rect, num_frames=128):
    print("Zoom test")
    with torch.no_grad():
        imgs = []
        for i in tqdm(range(num_frames)):
            pct = (i/(num_frames-1))
            x_min = start_rect[0]*(1-pct) + end_rect[0]*pct
            x_max = start_rect[1]*(1-pct) + end_rect[1]*pct
            y_min = start_rect[2]*(1-pct) + end_rect[2]*pct
            y_max = start_rect[3]*(1-pct) + end_rect[3]*pct
            g = [torch.linspace(x_min, x_max, res[0], device=model.device),
                torch.linspace(y_min, y_max, res[1], device=model.device)]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            out = model(g).reshape(res+[model.n_channels])#.flip(dims=[0,1])
            imgs.append(to_img(out))
    return imgs

def zoom_test_density(model, res, start_rect, end_rect, num_frames=128):
    print("Zoom test density")
    with torch.no_grad():
        imgs = []
        for i in tqdm(range(num_frames)):
            pct = (i/(num_frames-1))
            x_min = start_rect[0]*(1-pct) + end_rect[0]*pct
            x_max = start_rect[1]*(1-pct) + end_rect[1]*pct
            y_min = start_rect[2]*(1-pct) + end_rect[2]*pct
            y_max = start_rect[3]*(1-pct) + end_rect[3]*pct
            g = [torch.linspace(x_min, x_max, res[0], device=model.device),
                torch.linspace(y_min, y_max, res[1], device=model.device)]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            out = model.vis_heatmap(g).reshape(res)#.flip(dims=[0,1])
            imgs.append(to_img(out))
    return imgs

def throughout_test(model, batch_size=2**20, num_iters=50):
    with torch.no_grad():
        model.train(False)
        x = torch.rand([batch_size, model.opt['num_dims']], dtype=torch.float32, device=model.opt['device'])
        t0 = time.time()
        for i in range(num_iters):
            model(x)
        t1 = time.time()
        elapsed = t1-t0
    
    print(f"Throughput: {num_iters*batch_size / elapsed} points per second")

def profiler_test(model):
    x = torch.rand([2**20, model.opt['num_dims']], dtype=torch.float32, device=model.opt['device'])
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        y = model.forward(x)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=10))


if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    img_name = "pluto.png"
    device = "cuda"
    training_img = load_img(os.path.join(data_folder, img_name))
    img_shape = list(training_img.shape)[0:2]

    model_name = "pluto_1000000_gaussians"
    opt = load_options(os.path.join(save_folder, 
                       model_name))
    dataset = create_dataset(opt)
    model = load_model(opt)

    g_x = torch.arange(0, img_shape[0], dtype=torch.float32, device=device) / (img_shape[0]-1)
    g_y = torch.arange(0, img_shape[1], dtype=torch.float32, device=device) / (img_shape[1]-1)
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)

    x = training_img_positions
    

    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values
    g = [torch.linspace(xmin[i], xmax[i], img_shape[i], device=device) for i in range(xmin.shape[0])]
    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
    
    profiler_test(model)
    throughout_test(model)
    '''
    start = [xmin[0], xmax[0], xmin[1], xmax[1]]
    #end = [0.15, 0.25, 0.85, 0.95]
    end = [0.45, 0.55, 0.45, 0.55]

    imgs = zoom_test(model, img_shape, start, end)
    imageio.imwrite(os.path.join("./output", f"{model_name}_zoom.gif"), imgs)

    imgs = zoom_test_density(model, img_shape, start, end)
    imageio.imwrite(os.path.join("./output", f"{model_name}_zoom_density.gif"), imgs)
    '''