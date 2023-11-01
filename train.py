
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['KINETO_LOG_LEVEL'] = '3'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
    writer.add_histogram("Inverse scale distribution", torch.exp(model.gaussian_scales.flatten()), i)

class ImageDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_path, batch_size=2**17, device="cuda"):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.batch_size = batch_size
        self.device = device
        training_img = load_img("./data/"+img_path)
        img_shape = list(training_img.shape)[0:2]
        self.og_img_shape = list(training_img.shape)
        
        g_x = torch.arange(0, img_shape[0], dtype=torch.float32, device=device) / (img_shape[0]-1)
        g_y = torch.arange(0, img_shape[1], dtype=torch.float32, device=device) / (img_shape[1]-1)
        training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                            dim=-1).reshape(-1, 2).type(torch.float32)
        training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=device).reshape(-1,training_img.shape[-1])

        self.x = training_img_positions
        self.y = training_img_colors 
        
        max_img_reconstruction_dim_size = 1024    
        xmin = self.x.min(dim=0).values
        xmax = self.x.max(dim=0).values

        img_scale = min(max_img_reconstruction_dim_size, max(img_shape))/max(img_shape)
        self.training_preview_img_shape = [int(img_shape[i]*img_scale) for i in range(xmin.shape[0])]
        g = [torch.linspace(xmin[i], xmax[i], self.training_preview_img_shape[i], device=self.device) for i in range(xmin.shape[0])]
        self.training_preview_positions = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)

        self.training_samples = torch.rand(len(self), device=self.device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if(self.batch_size < len(self)):
            sample = self.training_samples.random_().topk(self.batch_size, dim=0).indices
            return self.x[sample], self.y[sample]
        else:
            return self.x, self.y
    
if __name__ == '__main__':
    
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Trains an implicit model on data.')

    parser.add_argument('--n_dims',default=2,type=int,
        help='Number of dimensions in the data')
    parser.add_argument('--n_outputs',default=3,type=int,
        help='Number of output channels for the data (ex. 1 for scalar field, 3 for image or vector field)')
    parser.add_argument('--n_gaussians',default=100000,type=int,
        help='Number of gaussians to use')
    parser.add_argument('--n_starting_gaussians',default=2560,type=int,
        help='Number of gaussians to use at start')
    parser.add_argument('--gaussian_only',default="False",type=str2bool,
        help='Whether to use only gaussians or include the waveform part.')  
    parser.add_argument('--max_frequency',default=128.,type=float,
        help='Maximum frequency for a primitive.') 
    parser.add_argument('--num_frequencies',default=128,type=int,
        help='Total number of frequencies per primitive.') 
    parser.add_argument('--num_top_frequencies',default=4,type=int,
        help='Num top frequencies used in model.') 
    parser.add_argument('--num_random_frequencies',default=0,type=int,
        help='Num random frequencies used in model.') 
    parser.add_argument('--data',default=None,type=str,
        help='Data file name')
    parser.add_argument('--save_name',default=None,type=str,
        help='Save name for the model')
    parser.add_argument('--batch_size',default=2**17,type=int,
        help='Batch size per update')  
    parser.add_argument('--train_iters',default=30000,type=int,
        help='Number of iterations to train total')    
    parser.add_argument('--fine_tune_iters',default=5000,type=int,
        help='Number of iterations to fine tune')    
    parser.add_argument('--split_iters',default=1000,type=int,
        help='Iterations between gaussian splits')     
    parser.add_argument('--prune_iters',default=100,type=int,
        help='Iterations between gaussian pruning')  
    parser.add_argument('--black_out_iters',default=3000,type=int,
        help='Iterations between blackouts')  
    parser.add_argument('--device',default="cuda:0",type=str,
        help='Which device to train on')
    args = vars(parser.parse_args())


    total_iters = args['train_iters']
    fine_tune_iters = args['fine_tune_iters']  
    starting_primitives = args['n_starting_gaussians']
    total_primitives = args['n_gaussians']
    only_gaussians = args['gaussian_only']
    
    split_every = args['split_iters']
    prune_every = args['prune_iters']
    black_out_every = args['black_out_iters']
    batch_size = args['batch_size']
    img_name = args['data']
    device = args['device']

    if(args['save_name'] is None):
        current_t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_name = img_name.split('.')[0]+"_"+current_t
    else:
        save_name = args['save_name']

    
    data = ImageDataset(args['data'], batch_size=batch_size, device=device)
    min_radius = 8*max(data.og_img_shape[0], data.og_img_shape[1])
    model = PeriodicPrimitives2D(device=device, n_channels=args['n_outputs'], 
                                 gaussian_only=only_gaussians, total_iters=total_iters,
                                 max_frequency=args['max_frequency'],
                                 num_frequencies=args['num_frequencies'],
                                 num_top_freqs=args['num_top_frequencies'],
                                 num_random_freqs=args['num_random_frequencies'],
                                 )

    n_extracted_peaks = model.add_primitives(starting_primitives)
    
    writer = SummaryWriter(f"./runs/{save_name}")
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
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./runs/traces/{save_name}"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,

    ) as prof:
        for i in t:
            x, y = data[i]
            # image logging
            if i % 1000 == 0 and i > 0:
                log_imgs(writer, model, data.training_preview_positions, data.training_preview_img_shape, i)
            # histogram logging
            if i % 1000 == 0 and i > 0:
                log_frequencies(writer, model, i)

            model.update_learning_rate(i)

            # Prune primitives
            if i % prune_every == 0 and i > 0 and prune_every > 0:
                with torch.no_grad():
                    prims_removed = model.prune_primitives(min_contribution=1./1000., 
                                                           min_width=min_radius)
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
            losses, model_out = model.loss(x, y)
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
        output = torch.empty([data.x.shape[0], model.n_channels], dtype=torch.float32, device=model.device)
        max_batch = int((2**28) / 256.)
        for i in range(0, data.x.shape[0], max_batch):
            end = min(output.shape[0], i+max_batch)
            output[i:end] = model(data.x[i:end])
        final_im = to_img(output.reshape(data.og_img_shape))
        print(final_im.shape)
        imageio.imwrite(os.path.join("./output", save_name+".jpg"), final_im)
        
        p = psnr(output,data.y).item()
        ssim_func = ssim(data_range=1.0).to(model.device)
        lpips_func = lpips().to(model.device)
        s = ssim_func(output.reshape(data.og_img_shape).permute(2,0,1)[None,...], 
                      data.y.reshape(data.og_img_shape).permute(2,0,1)[None,...])
        l = lpips_func(output.reshape(data.og_img_shape).permute(2,0,1)[None,...], 
                      data.y.reshape(data.og_img_shape).permute(2,0,1)[None,...])
        print(f"Final PSNR: {p:0.02f}")
        print(f"Final SSIM: {s:0.04f}")
        print(f"Final LPIPS: {l:0.04f}")
        writer.add_scalar("Params vs. PSNR", 
                        p, model.param_count())
        writer.add_scalar("Effective params vs. PSNR", 
                        p, model.effective_param_count())
    writer.flush()
    writer.close()
    model.save(os.path.join("./savedModels", save_name+".ckpt"))
    
