
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
from datasets.ImageDataset import make_coord_grid
from datasets.datasets import create_dataset
from models.models import create_model, load_model
from models.options import Options, load_options, save_options, update_options_from_args
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

project_folder_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def zoom_test(model, res, start_rect, end_rect, num_frames=128):

    with torch.no_grad():
        imgs = []
        for i in tqdm(range(num_frames)):
            pct = ((i+1)/(num_frames))
            x_min = start_rect[0]*(1-pct) + end_rect[0]*pct
            x_max = start_rect[1]*(1-pct) + end_rect[1]*pct
            y_min = start_rect[2]*(1-pct) + end_rect[2]*pct
            y_max = start_rect[3]*(1-pct) + end_rect[3]*pct
            g = [torch.linspace(x_min, x_max, res[0], device=model.opt['device']),
                torch.linspace(y_min, y_max, res[1], device=model.opt['device'])]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            out = model.forward(g).reshape(res+[3])
            im = to_img(out).swapaxes(0,1)
            imgs.append(im)
    return imgs

def zoom_psnr(model, dataset, res, rect):
     with torch.no_grad():
        g = [torch.linspace(rect[0], rect[1], res[0], device=model.opt['device']),
            torch.linspace(rect[2], rect[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out_gt = dataset.forward(g).reshape(res+[3])
        out_model = model.forward(g).reshape(res+[3])
        p = psnr(out_model, out_gt)
        print(f"Zoom PSNR: {p:0.02f}")

def zoom_test_vis_kernels(model, res, start_rect, end_rect, num_frames=128):

    with torch.no_grad():
        imgs = []
        for i in tqdm(range(num_frames)):
            pct = ((i+1)/num_frames)
            x_min = start_rect[0]*(1-pct) + end_rect[0]*pct
            x_max = start_rect[1]*(1-pct) + end_rect[1]*pct
            y_min = start_rect[2]*(1-pct) + end_rect[2]*pct
            y_max = start_rect[3]*(1-pct) + end_rect[3]*pct
            g = [torch.linspace(x_min, x_max, res[0], device=model.opt['device']),
                torch.linspace(y_min, y_max, res[1], device=model.opt['device'])]
            g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
            out = model.vis_kernel(g).reshape(res+[4])
            im = to_img(out).swapaxes(0,1)
            imgs.append(im)
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

def quant_metrics(model, dataset):
    with torch.no_grad():
        output = torch.empty(dataset.shape(), 
                             dtype=torch.float32, 
                             device=opt['data_device']).flatten(0,1)

        max_batch = int(2**28 / 64)
        points = (make_coord_grid([dataset.shape()[0], dataset.shape()[1]], 
            device=opt['data_device'], align_corners=True) + 1.)/2.
        for i in tqdm(range(0, points.shape[0], max_batch), "Full reconstruction"):
            end = min(points.shape[0], i+max_batch)
            output[i:end] = model(points[i:end].to(opt['device'])).to(opt['data_device'])
        output = output.reshape(dataset.shape())
        del points
        p = psnr(output.to(opt['data_device']),dataset.img).item()
        print(f"Final PSNR: {p:0.02f}")
        torch.cuda.empty_cache()

        ssim_func = ssim(data_range=1.0).to(opt['device'])        
        lpips_func = lpips().to(opt['device'])        

        output = output.to(opt['device']).reshape(dataset.shape()).permute(2,0,1)[None,...]
        dataset.img = dataset.img.permute(2,0,1)[None,...]

        iters = 0
        ssim_sum = 0.
        lpips_sum = 0.

        for y in range(0, output.shape[2], 2048):
            y_max = min(output.shape[2], y+2048)
            for x in range(0, output.shape[3], 2048):
                x_max = min(output.shape[3], x+2048)

                output_batch = output[:,:,y:y_max,x:x_max].to(opt['device'])
                img_batch = dataset.img[:,:,y:y_max,x:x_max].to(opt['device'])
                ssim_sum += ssim_func(output_batch, img_batch)
                lpips_sum += lpips_func(output_batch, img_batch)
                iters += 1
            
        print(f"Final SSIM: {ssim_sum/iters:0.04f}")
        print(f"Final LPIPS: {lpips_sum/iters:0.04f}")
        torch.cuda.empty_cache()

def save_output(model, dataset):
    with torch.no_grad():
        output = torch.empty(dataset.shape(), 
                             dtype=torch.float32, 
                             device=opt['data_device']).flatten(0,1)

        max_batch = int(2**28 / 64)
        points = (make_coord_grid([dataset.shape()[0], dataset.shape()[1]], 
            device=opt['data_device'], align_corners=True) + 1.)/2.
        for i in tqdm(range(0, points.shape[0], max_batch), "Full reconstruction"):
            end = min(points.shape[0], i+max_batch)
            output[i:end] = model(points[i:end].to(opt['device'])).to(opt['data_device'])
        output = output.reshape(dataset.shape())
        del points
        p = psnr(output.to(opt['data_device']),dataset.img).item()
        print(f"Final PSNR: {p:0.02f}")
        torch.cuda.empty_cache()

        ssim_func = ssim(data_range=1.0).to(opt['device'])        
        lpips_func = lpips().to(opt['device'])        

        output = output.to(opt['device']).reshape(dataset.shape()).permute(2,0,1)[None,...]
        dataset.img = dataset.img.permute(2,0,1)[None,...]

        iters = 0
        ssim_sum = 0.
        lpips_sum = 0.

        for y in range(0, output.shape[2], 2048):
            y_max = min(output.shape[2], y+2048)
            for x in range(0, output.shape[3], 2048):
                x_max = min(output.shape[3], x+2048)

                output_batch = output[:,:,y:y_max,x:x_max].to(opt['device'])
                img_batch = dataset.img[:,:,y:y_max,x:x_max].to(opt['device'])
                ssim_sum += ssim_func(output_batch, img_batch)
                lpips_sum += lpips_func(output_batch, img_batch)
                iters += 1
            
        print(f"Final SSIM: {ssim_sum/iters:0.04f}")
        print(f"Final LPIPS: {lpips_sum/iters:0.04f}")
        torch.cuda.empty_cache()

def count_params(model):
    print(model.effective_param_count())

def create_aaselund_zoom(model, model_name):
    res = [512,512]
    r = 5769./14586.
    frames = [
        [0.84-0.1*r/2, 0.84+0.1*r/2, 0.75, 0.85],
        [0.285-0.1*r/2, 0.285+0.1*r/2, 0.75, 0.85],
        [0.4-0.1*r/2, 0.4+0.1*r/2, 0.6, 0.7],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_denman_zoom(model, model_name):
    res = [512,512]
    r = 3207./30409.
    frames = [
        [0.84-0.2*r/2, 0.84+0.2*r/2, 0.70, 0.9],
        [0.285-0.2*r/2, 0.285+0.2*r/2, 0.77, 0.9],
        [0.6-0.2*r/2, 0.6+0.2*r/2, 0.55, 0.75],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_mtcook_zoom(model, model_name):
    res = [512,512]
    r = 6566./15490.
    frames = [
        [0.84-0.2*r/2, 0.84+0.2*r/2, 0.70, 0.9],
        [0.4-0.1*r/2, 0.4+0.1*r/2, 0.2, 0.3],
        [0.6-0.1*r/2, 0.6+0.1*r/2, 0.4, 0.5],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_norway_zoom(model, model_name):
    res = [512,512]
    r = 5421./18904.
    frames = [
        [0.25-0.2*r/2, 0.25+0.2*r/2, 0.5, 0.7],
        [0.7-0.1*r/2, 0.7+0.1*r/2, 0.6, 0.7],
        [0.97-0.1*r/2, 0.97+0.1*r/2, 0.45, 0.55],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_vermont_zoom(model, model_name):
    res = [512,512]
    r = 5179./36931.
    frames = [
        [0.25-0.2*r/2, 0.25+0.2*r/2, 0.5, 0.7],
        [0.68-0.1*r/2, 0.68+0.1*r/2, 0.2, 0.3],
        [0.97-0.1*r/2, 0.97+0.1*r/2, 0.45, 0.55],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_tokyo_zoom(model, model_name):
    res = [512,512]
    r = 21450./56718.
    frames = [
        [0.84-0.1*r/2, 0.84+0.1*r/2, 0.2, 0.3],
        [0.595-0.1*r/2, 0.595+0.1*r/2, 0.25, 0.35],
        [0.5-0.1*r/2, 0.5+0.1*r/2, 0.72, 0.82],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_pluto_zoom(model, model_name):
    res = [512,512]
    r = 1.
    frames = [
        [0.84-0.05*r/2, 0.84+0.05*r/2, 0.25, 0.3],
        [0.595-0.05*r/2, 0.595+0.05*r/2, 0.23, 0.35],
        [0.5-0.05*r/2, 0.5+0.05*r/2, 0.77, 0.82],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_lighthouse_zoom(model, model_name):
    res = [512,512]
    r = 8708/11608.
    frames = [
        [0.20-0.1*r/2, 0.20+0.1*r/2, 0.45, 0.55],
        [0.5-0.1*r/2, 0.5+0.1*r/2, 0.27, 0.37],
        [0.72-0.1*r/2, 0.72+0.1*r/2, 0.62, 0.72],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_earring_zoom(model, model_name):
    res = [512,512]
    r = 36000/50000.

    frames = [
        [0.70-0.03/2, 0.70+0.03/2, 0.04-0.03*r/2, 0.04+0.03*r/2],
        [0.60-0.03/2, 0.60+0.03/2, 0.35-0.03*r/2, 0.35+0.03*r/2],
        [0.72-0.03/2, 0.72+0.03/2, 0.61-0.03*r/2, 0.61+0.03*r/2],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

def create_starbirth_zoom(model, model_name):
    res = [512,512]
    r = 11710/17043.

    frames = [
        [0.70-0.1*r/2, 0.70+0.1*r/2, 0.5-0.1/2, 0.2+0.1/2],
        [0.60-0.1*r/2, 0.60+0.1*r/2, 0.35-0.1/2, 0.35+0.1/2],
        [0.2-0.1*r/2, 0.2+0.1*r/2, 0.8-0.1/2, 0.8+0.1/2],
    ]
    i = 0
    for frame in frames:
        g = [torch.linspace(frame[0], frame[1], res[0], device=model.opt['device']),
            torch.linspace(frame[2], frame[3], res[1], device=model.opt['device'])]
        g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
        out = model.forward(g).reshape(res+[3])
        im = to_img(out).swapaxes(0,1)
        dest = os.path.join("./output", f"{model_name}_{i}.png")
        imageio.imwrite(dest, im)
        i += 1

if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    #img_name = "pluto.png"
    #device = "cuda"
    #training_img = load_img(os.path.join(data_folder, img_name))
    #img_shape = list(training_img.shape)[0:2]

    model_name = "Starbirth_iNGP"
    location = os.path.join(save_folder, 
                       model_name)
    opt = load_options(location)
    opt['data_device'] = "cuda:0"
    model = load_model(opt,location)
    #quant_metrics(model, dataset)

    #dataset = create_dataset(opt)
    #model_name = "GirlWithPearlEarring_GT"

    #start_pluto, end_pluto = [0.0, 1.0, 0.0, 1.0], [0.67, 0.685, 0.36, 0.375]
    #r = 0.3781
    #start_tokyo, end_tokyo = [0.5-r/2, 0.5+r/2, 0, 1.], [0.3-0.06*r/2, 0.3+0.06*r/2, 0.33, 0.39]
    #r = 11608./8708.
    #start_lighthouse, end_lighthouse = [0.5-r/2, 0.5+r/2, 0, 1.], [0.53-0.04*r/2, 0.53+0.04*r/2, 0.48, 0.52]
    #r = 6566./15490.
    #start_mtcook, end_mtcook = [0.5-r/2, 0.5+r/2, 0, 1.], [0.36-0.07*r/2, 0.36+0.07*r/2, 0.26, 0.33]
    #r = 5769./14586.
    #start_aalesund, end_aalesund = [0.5-r/2, 0.5+r/2, 0, 1.], [0.84-0.1*r/2, 0.84+0.1*r/2, 0.75, 0.85]
    #r = 36000./50000.
    #start_pearl, end_pearl = [0, 1., 0.5-r/2, 0.5+r/2], [0.77, 0.83, 0.84-0.06*r/2, 0.84+0.06*r/2]

    #imgs = zoom_test(model, [2048,1024], [0.18-(0.1*r/2), 0.18+(0.1*r/2), 0.45, 0.55], [0.18-(0.1*r/2), 0.18+(0.1*r/2), 0.45, 0.55], num_frames=1)
    #imgs = zoom_test(dataset, [1024,1024], [0, 1, 0, 1], [0, 1, 0, 1], num_frames=1)
    #dest = os.path.join("./output", f"{model_name}_GT.png")
    #imageio.imwrite(dest, imgs[0], fps=30, quality=10)
    #zoom_psnr(model, dataset, [2048,1024], [0.18-(0.1*r/2), 0.18+(0.1*r/2), 0.45, 0.55])
    
    #model = create_dataset(opt)
    #model_name = "Starbirth_GT"
    create_starbirth_zoom(model, model_name)