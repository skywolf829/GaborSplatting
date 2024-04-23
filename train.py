
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as lpips
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
from torch.utils.data import DataLoader
import json
from datasets.ImageDataset import make_coord_grid
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

def log_imgs(writer, model, i, resolution=[768, 768], device="cuda"):
    with torch.no_grad():     
        points = (make_coord_grid(resolution, device=device, align_corners=True)+1)/2
        img = model(points).reshape(resolution[0], resolution[1], -1)
        img = to_img(img)
        writer.add_image('reconstruction', img, i, dataformats='HWC')
        heatmap = model.vis_heatmap(points).reshape(resolution[0], resolution[1], -1)
        if heatmap is not None:
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
        
    dataloader = DataLoader(dataset, batch_size=None, num_workers=8 if "cpu" in opt['data_device'] else 0)
    dataloader = iter(dataloader)
    torch.cuda.empty_cache()
    max_VRAM = torch.cuda.max_memory_allocated(opt['device'])
    torch.cuda.synchronize()
    start_train_time = time.time()

    # Train model
    for i in t:
        model.training_routine_updates(i, writer=writer)

        x, y = next(dataloader)
        model.optimizer.zero_grad()
        losses, model_out = model.loss(x, y)
        losses['final_loss'].backward()
        if i < opt['fine_tune_iterations']:
            model.update_cumulative_gradients()
        model.optimizer.step()
            
        # logging
        if opt['log_every'] > 0 and i % opt['log_every'] == 0:
            p = log_scalars(writer, model, losses, i)
            t.set_description(f"[{i+1}/{opt['train_iterations']}] PSNR: {p:0.04f}")
        # image logging
        if opt['log_image_every'] > 0 and i % opt['log_image_every'] == 0 and i > 0:
            log_imgs(writer, model, i)
            log_frequencies(writer, model, i)
        
        if(opt['profile']):
            profiler.step()
        max_VRAM = max(max_VRAM, torch.cuda.max_memory_allocated(opt['device']))

    dataloader = None
    del dataloader
    torch.cuda.synchronize()
    end_train_time = time.time()
    total_train_time = int(end_train_time - start_train_time)
    max_VRAM_MB = int(max_VRAM/1000000)
    print(f"Total train time: {int(total_train_time/60)}m {int(total_train_time%60)}s")
    print(f"Max VRAM: {max_VRAM_MB}MB")
    if not os.path.exists(os.path.join(save_folder, save_name)):
        try:
            os.makedirs(os.path.join(save_folder, save_name))
        except OSError:
            print(f"Creation of the directory {os.path.join(save_folder, save_name)} failed")
    save_options(opt, os.path.join(save_folder, save_name))
    model.save(os.path.join(save_folder, save_name))
    torch.cuda.empty_cache()

    # Test model
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

        #print("Saving image...")
        #final_im = to_img(output)
        #imageio.imwrite(os.path.join(output_folder, opt['save_name']+".png"), final_im)
        
        final_results = {}
        #err = ((output.to(opt['data_device']) - dataset.img)**2)
        #print("Saving error map...")
        #err_img = to_img(err.reshape(dataset.shape()))
        #del err
        #imageio.imwrite(os.path.join(output_folder, f"{save_name}_err.png"), err_img)
        torch.cuda.empty_cache()

        print("Computing metrics:")
        try:
            p = psnr(output.to(opt['data_device']),dataset.img).item()
            print(f"Final PSNR: {p:0.02f}")
            final_results["PSNR"] = p
            torch.cuda.empty_cache()
            writer.add_scalar("Params vs. PSNR", 
                            p, model.param_count())
            writer.add_scalar("Effective params vs. PSNR", 
                            p, model.effective_param_count())
            
            writer.flush()
            writer.close()

            ssim_func = ssim(data_range=1.0).to(opt['device'])        
            lpips_func = lpips().to(opt['device'])        

            output = output.reshape(dataset.shape()).permute(2,0,1)[None,...]
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
        except:
            print("Error caught, likely OOM")
        
        print(f"Final SSIM: {ssim_sum/iters:0.03f}")
        print(f"Final LPIPS: {lpips_sum/iters:0.03f}")

        total_params = model.param_count()
        final_results['num_params'] = total_params
        print(f"Total num params: {total_params:,}")
        final_results['train_time'] = total_train_time
        final_results['VRAM_MB'] = max_VRAM_MB
        torch.cuda.empty_cache()
        with open(os.path.join(save_folder, save_name, "results.json"), 'w') as fp:
            json.dump(final_results, fp, indent=4)

if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')
    parser.add_argument('--model',default=None,type=str,
        help='Model type. "Splats" or "iNGP"')
    parser.add_argument('--num_outputs',default=None,type=int,
        help='Number of output channels for the data (ex. 1 for grayscale, 3 for RGB)')
    parser.add_argument('--num_total_prims',default=None,type=int,
        help='Number of gaussians to reach by end of training')
    parser.add_argument('--num_starting_prims',default=None,type=int,
        help='Number of gaussians to use at start')
    parser.add_argument('--gaussian_only',default=None,type=str2bool,
        help='Whether to use only gaussians. False uses gabor.')  
    parser.add_argument('--max_frequency',default=None,type=float,
        help='Maximum frequency for a primitive in Hz.') 
    parser.add_argument('--num_total_frequencies',default=None,type=int,
        help='Total number of frequencies per primitive. Filter bank size.') 
    parser.add_argument('--num_frequencies',default=None,type=int,
        help='Num top frequencies used in model. k value from paper.') 
    parser.add_argument('--training_data',default=None,type=str,
        help='Data file name, assumed to be in data/ folder.')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model. Creates a folder in savedModels/ of this name')
    parser.add_argument('--batch_size',default=None,type=int,
        help='Batch size per step.')  
    parser.add_argument('--train_iterations',default=None,type=int,
        help='Number of iterations to train total')    
    parser.add_argument('--fine_tune_iterations',default=None,type=int,
        help='Number of iterations to fine tune. Gaussians will stop splitting when there are this many iterations left.')    
    parser.add_argument('--split_every_iters',default=None,type=int,
        help='Iterations between gaussian splits')     
    parser.add_argument('--prune_every_iters',default=None,type=int,
        help='Iterations between gaussian pruning')  
    parser.add_argument('--blackout_every_iters',default=None,type=int,
        help='Iterations between blackouts (sets color values near zero)')  
    parser.add_argument('--device',default=None,type=str,
        help='Which device to train on/where the model resides')
    parser.add_argument('--data_device',default=None,type=str,
        help='Which device to host training data on. Useful for very large images (gigapixel)')
    parser.add_argument('--load_from',default=None,type=str,
        help='Where to load model data from')
    parser.add_argument('--log_every',default=None,type=int,
        help='How often to log metrics')  
    parser.add_argument('--log_image_every',default=None,type=int,
        help='How often to log image.')  
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

    
    
