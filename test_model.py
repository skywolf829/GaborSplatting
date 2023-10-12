
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    model = PeriodicPrimitives2D()
    img_name = "truck.jpg"
    device = "cuda"
    training_img = load_img("./data/"+img_name)
    img_shape = list(training_img.shape)[0:2]

    #model_name = "PeriodicPrimitives2D_truck2023-10-12_13-27-53" # gaussians
    model_name = "PeriodicPrimitives2D_truck_2023-10-12_14-41-48" # periodic primitives

    model.load(f"./savedModels/{model_name}.ckpt")

    g_x = torch.arange(0, img_shape[0], dtype=torch.float32, device=device) / (img_shape[0]-1)
    g_y = torch.arange(0, img_shape[1], dtype=torch.float32, device=device) / (img_shape[1]-1)
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)

    x = training_img_positions
    

    xmin = x.min(dim=0).values
    xmax = x.max(dim=0).values
    g = [torch.linspace(xmin[i], xmax[i], img_shape[i], device=model.device) for i in range(xmin.shape[0])]
    g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
    
    start = [xmin[0], xmax[0], xmin[1], xmax[1]]
    #end = [0.15, 0.25, 0.85, 0.95]
    end = [0.45, 0.55, 0.45, 0.55]

    imgs = zoom_test(model, img_shape, start, end)
    imageio.imwrite(os.path.join("./output", f"{model_name}_zoom.gif"), imgs)

    imgs = zoom_test_density(model, img_shape, start, end)
    imageio.imwrite(os.path.join("./output", f"{model_name}_zoom_density.gif"), imgs)