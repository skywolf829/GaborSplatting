#
# This file tests that the CUDA kernels work as expected, compared with built-in PyTorch.
# A forward pass and backward pass is tested with multiple configurations for boundary conditions.
#

import torch
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
import argparse
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips
from datasets.ImageDataset import make_coord_grid
from models.models import create_model, load_model
from models.options import Options, load_options, save_options, update_options_from_args
from utils.data_utils import str2bool
from utils.data_utils import to_img, psnr
from tqdm import tqdm
import numpy as np
from datasets.datasets import create_dataset
import os
import imageio.v3 as imageio
import matplotlib.pyplot as plt
#plt.style.use("seaborn-paper")
plt.rcParams.update({'font.size': 16})

project_folder_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def generate_full_im(model, dataset, opt):
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
    return output

def compute_metrics(output, dataset, opt):
    print("Computing metrics:")
    try:
        p = psnr(output.to(opt['data_device']),dataset.img).item()
        print(f"Final PSNR: {p:0.02f}")
        final_results["PSNR"] = p
        torch.cuda.empty_cache()
        
        if("cuda" in opt['data_device']):
            ssim_func = ssim(data_range=1.0).to(opt['data_device'])
            s = ssim_func(output.to(opt['data_device']).reshape(dataset.shape()).permute(2,0,1)[None,...], 
                dataset.img.permute(2,0,1)[None,...])
            print(f"Final SSIM: {s:0.04f}")
            final_results["SSIM"] = s.item()
            torch.cuda.empty_cache()
            lpips_func = lpips().to(opt['data_device'])        
            l = lpips_func(output.to(opt['data_device']).reshape(dataset.shape()).permute(2,0,1)[None,...], 
                dataset.img.permute(2,0,1)[None,...])
            print(f"Final LPIPS: {l:0.04f}")
            final_results["LPIPS"] = l.item()
    except:
        print("Error caught, likely OOM")

def frequency_distribution(model):
    indices = model.get_topk_waves()[1].detach().cpu().numpy().flatten()
    #indices = indices.astype(np.float32) * model.opt['max_frequency'] / (model.opt['num_total_frequencies']-1)
    counts, _ = np.histogram(indices, bins=-0.5+np.arange(0, model.opt['num_total_frequencies']+1), density = True)
    
    
    plt.bar(np.arange(0, model.opt['num_total_frequencies'])* model.opt['max_frequency'] / (model.opt['num_total_frequencies']-1), 
            counts,
            width = model.opt['max_frequency'] / (model.opt['num_total_frequencies']-1) / 1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Proportion")
    plt.title(f"Frequency distribution for Tokyo")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')
    parser.add_argument('--load_from',default=None,type=str,
        help='Where to load model data from')
    parser.add_argument('--output_img',default=1,type=str2bool,
        help='Generate and save output image')
    parser.add_argument('--output_err_img',default=0,type=str2bool,
        help='Generate and save output error image')
    parser.add_argument('--output_frequency_distribution',default=0,type=str2bool,
        help='Generate chart of frequency distribution')
    parser.add_argument('--compute_metrics',default=0,type=str2bool,
        help='Where to load model data from')
    parser.add_argument('--device',default="cuda",type=str,
        help='Device to use.')
    args = vars(parser.parse_args())

    if args['load_from'] is None:
        print(f"Must enter a --load_from argument")
        quit()

    # Load options and model file
    opt = load_options(os.path.join(save_folder, 
                            args["load_from"]))
    opt["device"] = args["device"]
    update_options_from_args(opt, args)
    dataset = create_dataset(opt)
    model = load_model(opt, os.path.join(save_folder, 
                            args["load_from"]))

    if(args['output_frequency_distribution']):
        frequency_distribution(model)

    if(args['output_img'] or args['output_err_img'] or args['compute_metrics']):
        # Get model's output
        output = generate_full_im(model, dataset, opt)

        # Save images
        if args['output_img']:
            print("Saving image...")
            final_im = to_img(output)
            imageio.imwrite(os.path.join(output_folder, opt['save_name']+".png"), final_im)
        
        final_results = {}
        err = ((output.to(opt['data_device']) - dataset.img)**2)
        if args['output_err_img']:
            print("Saving error map...")
            err_img = to_img(err.reshape(dataset.shape()))
            del err
            imageio.imwrite(os.path.join(output_folder, f"{opt['save_name']}_err.png"), err_img)
        torch.cuda.empty_cache()

        # Compute metrics
        if args['compute_metrics']:
            compute_metrics(output, dataset, opt)