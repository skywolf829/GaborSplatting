
print("Loading HybridPrimitives CUDA kernel. May need to compile...")
#from models.HybridPrimitives import HybridPrimitives
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
print("Successfully loaded HybridPrimitives.")
#from models.Siren import Siren
from utils.data_generators import load_img
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imageio.v3 as imageio
from tqdm import tqdm
from utils.data_utils import to_img, psnr
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from utils.data_utils import str2bool
from datasets.datasets import create_dataset
from models.models import create_model, load_model
from models.options import Options, load_options, save_options, update_options_from_args
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KINETO_LOG_LEVEL'] = '3'


project_folder_path = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(project_folder_path, "data")
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

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

def log_imgs(writer, model, dataset, g, img_shape, i):
    with torch.no_grad():     
        img = model(g).reshape(img_shape[0], img_shape[1], -1)
        img = to_img(img)
        writer.add_image('reconstruction', img, i, dataformats='HWC')
        heatmap = model.vis_heatmap(g).reshape(img_shape[0], img_shape[1], -1)
        heatmap = to_img(heatmap)
        writer.add_image('heatmap', heatmap, i, dataformats='HWC')
       
def log_frequencies(writer, model, i):
    if(not model.opt['gaussian_only']):
        freqs, coeffs = model.get_weighed_frequency_dist()
        writer.add_histogram("Frequency distribution", freqs, i)
        writer.add_histogram("Frequency coeffs", coeffs, i)

    writer.add_histogram("Inverse scale distribution", torch.exp(model.gaussian_scales.flatten()), i)

def train_model(model, dataset, opt):
    
    writer = SummaryWriter(f"./runs/{save_name}")
    t = tqdm(range(opt['train_iterations']))
    if(opt['profile']):
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=10,
                active=10,
                repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./runs/traces/{save_name}"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
        ) 
        
    for i in t:
        model.training_routine_updates(i, writer=writer)

        x, y = dataset[i]

        
        model.optimizer.zero_grad()
        losses, model_out = model.loss(x, y)
        losses['final_loss'].backward()
        model.update_cumulative_gradients()
        model.optimizer.step()
            
        # logging
        if i % opt['log_every'] == 0:
            p = log_scalars(writer, model, losses, i)
            t.set_description(f"[{i+1}/{opt['train_iterations']}] PSNR: {p:0.04f}")
        # image logging
        if i % opt['log_image_every'] == 0 and i > 0:
            log_imgs(writer, model, dataset, 
                     dataset.training_preview_positions.to(opt['device']), 
                     dataset.training_preview_img_shape, i)
            log_frequencies(writer, model, i)
        
        if(opt['profile']):
            profiler.step()

    if not os.path.exists(os.path.join(save_folder, save_name)):
        try:
            os.makedirs(os.path.join(save_folder, save_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(save_folder, save_name)} failed")
    save_options(opt, os.path.join(save_folder, save_name))
    model.save(os.path.join(save_folder, save_name))

    with torch.no_grad():
        output = torch.empty(dataset.get_output_shape(), 
                             dtype=torch.float32, 
                             device=opt['data_device']).flatten(0,1)

        max_batch = int((2**28) / 256.)
        for i in range(0, len(dataset), max_batch):
            end = min(output.shape[0], i+max_batch)
            output[i:end] = model(dataset.x[i:end].to(opt['device'])).to(opt['data_device'])
        final_im = to_img(output.reshape(dataset.get_output_shape()))
        imageio.imwrite(os.path.join(output_folder, opt['save_name']+".jpg"), final_im)
        
        p = psnr(output.to(opt['data_device']),dataset.y).item()
        print(f"Final PSNR: {p:0.02f}")
        writer.add_scalar("Params vs. PSNR", 
                        p, model.param_count())
        writer.add_scalar("Effective params vs. PSNR", 
                        p, model.effective_param_count())
        
        writer.flush()
        writer.close()

        ssim_func = ssim(data_range=1.0).to(opt['data_device'])
        s = ssim_func(output.to(opt['data_device']).reshape(dataset.get_output_shape()).permute(2,0,1)[None,...], 
            dataset.y.reshape(dataset.get_output_shape()).permute(2,0,1)[None,...])
        print(f"Final SSIM: {s:0.04f}")
        lpips_func = lpips().to(opt['data_device'])        
        l = lpips_func(output.to(opt['data_device']).reshape(dataset.get_output_shape()).permute(2,0,1)[None,...], 
            dataset.y.reshape(dataset.get_output_shape()).permute(2,0,1)[None,...])
        print(f"Final LPIPS: {l:0.04f}")
        


if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')

    parser.add_argument('--num_dims',default=None,type=int,
        help='Number of dimensions in the data')
    parser.add_argument('--num_outputs',default=None,type=int,
        help='Number of output channels for the data (ex. 1 for scalar field, 3 for image or vector field)')
    parser.add_argument('--num_total_prims',default=None,type=int,
        help='Number of gaussians to use')
    parser.add_argument('--num_starting_prims',default=None,type=int,
        help='Number of gaussians to use at start')
    parser.add_argument('--gaussian_only',default=None,type=str2bool,
        help='Whether to use only gaussians or include the waveform part.')  
    parser.add_argument('--max_frequency',default=None,type=float,
        help='Maximum frequency for a primitive.') 
    parser.add_argument('--num_total_frequencies',default=None,type=int,
        help='Total number of frequencies per primitive.') 
    parser.add_argument('--num_frequencies',default=None,type=int,
        help='Num top frequencies used in model.') 
    parser.add_argument('--training_data',default=None,type=str,
        help='Data file name')
    parser.add_argument('--training_data_type',default=None,type=str,
        help='Type of training data. Either "image" or "scene".')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model')
    parser.add_argument('--batch_size',default=None,type=int,
        help='Batch size per update')  
    parser.add_argument('--train_iterations',default=None,type=int,
        help='Number of iterations to train total')    
    parser.add_argument('--fine_tune_iterations',default=None,type=int,
        help='Number of iterations to fine tune')    
    parser.add_argument('--split_every_iters',default=None,type=int,
        help='Iterations between gaussian splits')     
    parser.add_argument('--prune_every_iters',default=None,type=int,
        help='Iterations between gaussian pruning')  
    parser.add_argument('--blackout_every_iters',default=None,type=int,
        help='Iterations between blackouts')  
    parser.add_argument('--device',default=None,type=str,
        help='Which device to train on')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to host training data on')
    parser.add_argument('--load_from',default=None,type=str,
        help='Where to load data from')
    args = vars(parser.parse_args())
    
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    if(args['load_from'] is not None):
        opt = load_options(os.path.join(save_folder, 
                            args["load_from"]))
        opt["device"] = args["device"]
        opt["save_name"] = args["load_from"]
        update_options_from_args(opt, args)
        dataset = create_dataset(opt)
        model = load_model(opt, opt['device'])
    else:
        opt = Options.get_default()
        update_options_from_args(opt, args)
        if(args['save_name'] is None):
            current_t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_name = opt['training_data'].split('.')[0]+"_"+current_t
        else:
            save_name = args['save_name']
        opt['save_name'] = save_name
        dataset = create_dataset(opt)
        model = create_model(opt)

    train_model(model, dataset, opt)

    
    
