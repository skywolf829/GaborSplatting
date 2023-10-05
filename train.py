from models.GaussianSplatting2D import GaussianSplatting2D
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
from models.HybridPrimitiveModel import HybridPrimitiveModel
from models.PeriodicGaussians2D import PeriodicGaussians2D
from models.PeriodicGaussianField import PeriodicGaussianField
from models.SupportedPeriodicPrimitives2D import SupportedPeriodicPrimitives2D
from models.PeriodicGaussians2Dfreqangle import PeriodicGaussians2Dfreqangle
print("Loading HybridPrimitives CUDA kernel. May need to compile...")
from models.HybridPrimitives import HybridPrimitives
print("Successfully loaded HybridPrimitives.")
from models.HaarPrimitives2D import HaarPrimitives2D
from models.Siren import Siren
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

if __name__ == '__main__':
    
    total_iters = 300000
    fine_tune_iters = 5000    
    total_primitives = 50000
    primitives_per_update = 50
    iters_per_primitive = int((total_iters-fine_tune_iters) / (total_primitives/primitives_per_update))
    start_freq = 20
    end_freq = 512

    model_type = HybridPrimitives
    img_name = "truck.jpg"

    device = "cuda"
    torch.random.manual_seed(42)
    np.random.seed(42)
    training_img = load_img("./data/"+img_name)
    training_img_copy = (training_img.copy() * 255).astype(np.uint8)
    og_img_shape = training_img.shape
    model = model_type(device=device, n_channels=3)

    g_x = torch.arange(0, og_img_shape[0], dtype=torch.float32, device=device) / (og_img_shape[0]-1)
    g_y = torch.arange(0, og_img_shape[1], dtype=torch.float32, device=device) / (og_img_shape[1]-1)
    training_img_positions = torch.stack(torch.meshgrid([g_x, g_y], indexing='ij'), 
                                        dim=-1).reshape(-1, 2).type(torch.float32)
    training_img_colors = torch.tensor(training_img, dtype=torch.float32, device=device).reshape(-1,training_img.shape[-1])

    if(False):
        m1 = training_img_colors[:,0]>0.5
        m2 = torch.rand([training_img_colors.shape[0]], device=m1.device)>0.99
        m1 *= m2
        training_img_positions = training_img_positions[m1]
        training_img_colors = training_img_colors[m1]
    # Table top
    if(False):
        aspect=og_img_shape[1]/og_img_shape[0]
        m1 = (training_img_positions[:,0] > aspect*0.35*training_img_positions[:,1] - 50/og_img_shape[0])
        m2 = (training_img_positions[:,0] < aspect*0.4*training_img_positions[:,1] + 760/og_img_shape[0])
        m3 = (training_img_positions[:,0] < aspect*-1.55*training_img_positions[:,1] + 2270/og_img_shape[0])
        m4 = (training_img_positions[:,1] < 1040/og_img_shape[1])
        # for top of table
        m5 = (training_img_positions[:,0] < aspect*0.375*training_img_positions[:,1] + 420/og_img_shape[0])
        m6 = (training_img_positions[:,0] < aspect*-1.25*training_img_positions[:,1] + 1620/og_img_shape[0])
        m7 = (training_img_positions[:,0] > aspect*-1.15*training_img_positions[:,1] + 700/og_img_shape[0])
        #training_img_positions = training_img_positions[m1*m2*m3*m4*m5*m6*m7]
        #training_img_colors = training_img_colors[m1*m2*m3*m4*m5*m6*m7]
        training_img_colors[~(m1*m2*m3*m4*m5*m6*m7)] = 0#training_img_colors[m1*m2*m3*m4*m5*m6].mean()
    #plt.scatter(training_img_positions[:,1].cpu().numpy(), -training_img_positions[:,0].cpu().numpy(), c=training_img_colors.cpu().numpy())
    #plt.show()

    x = training_img_positions
    y = training_img_colors 
    
    num_params = []

    pre_fitting_imgs = []
    wave_imgs = []
    max_tries = 5
    tries = max_tries

    max_ls_points = 2**17
    pct_of_data = max_ls_points / x.shape[0]
    writer = SummaryWriter(f"./runs/{img_name.split('.')[0]}/{model.__class__.__name__}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    t = tqdm(range(total_iters))
    for i in t:
        mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

        # image logging
        if i % 200 == 0 and i > 0:
            with torch.no_grad():
                res = [512, 512]
                xmin = x.min(dim=0).values
                xmax = x.max(dim=0).values
                g = [torch.linspace(xmin[i], xmax[i], res[i], device=model.device) for i in range(xmin.shape[0])]
                g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                img = model(g).reshape(res+[model.n_channels])
                img = to_img(img)
                writer.add_image('reconstruction', img, i, dataformats='HWC')
                
                #if("Siren" not in model.__class__.__name__):
                    #wave_img = to_img(model.vis_primitives(x))                
                    #writer.add_image('primitives', wave_img, i, dataformats='HWC')

        # adding primitives
        if i % iters_per_primitive == 0 and i < total_iters-fine_tune_iters and "Siren" not in model.__class__.__name__:
            with torch.no_grad():   
                if i > 0: 
                    writer.add_scalar("Params vs. PSNR", 
                                      20*np.log10(1.0) - 10*torch.log10(losses['mse']), 
                                      model.param_count())
                residuals = y[mask]
                if(i>0):
                    residuals -= model(x[mask])
                model.prune_primitives(1./500.)
                #n_gaussians = primitives_per_update
                #n_waves = 0
                n_gaussians = 0
                n_waves = primitives_per_update
                #n_gaussians = primitives_per_update-n_waves
                n_extracted_peaks = model.add_primitives(
                                    x[mask],
                                    residuals,
                                    n_freqs = 180, 
                                    #n_angles = 180,
                                    max_freq=(total_iters-i)*start_freq/total_iters+i*end_freq/total_iters, 
                                    min_influence=1./500.,
                                    num_waves = n_waves,
                                    num_gaussians = n_gaussians
                                    )
                if(n_waves > 0):
                    writer.add_scalar("Max LS power", 
                                        model.ls_power.max(), 
                                        i)
                    writer.add_scalar("Min LS power", 
                                        model.ls_power.min(), 
                                        i)
                
                if(n_waves > 0):
                    writer.add_image("Lomb-Scargle", model.ls_plot, i, dataformats="HWC")
        
        #if i % iters_per_wave == 0 and i > 0:
            #with torch.no_grad():
                #model.colors.zero_()

        # actual training step
        model.optimizer.zero_grad()
        losses, model_out = model.loss(x[mask], y[mask])
        #if (losses['final_loss'].isnan()):
        #    print()
        #    print(f"Detected loss was NaN.")
        #    quit()
            
        losses['final_loss'].backward()
        #print(f"{self.subgaussian_width} {self.subgaussian_width.grad}")
        model.optimizer.step()
        #with torch.no_grad():
        #    model.subgaussian_flat_top_power.clamp_(-1, 3)
            
        # logging
        if i % 100 == 0:
            with torch.no_grad():     
                writer.add_scalar("Loss", losses['final_loss'].item(), i)     
                p = 20*np.log10(1.0) - 10*torch.log10(losses['mse'])
                writer.add_scalar("Train PSNR", p, i)        
                t.set_description(f"[{i+1}/{total_iters}] PSNR: {p.item():0.04f}")
                writer.add_scalar("Num gaussians", model.get_num_gaussians(), i)     
                writer.add_scalar("Num waves", model.get_num_waves(), i)     
    #print(p.key_averages().table(
    #    sort_by="self_cuda_time_total", row_limit=-1))
    
    
    #imageio.imwrite("output/supported_waves_training_err.mp4", pre_fitting_imgs)
    #imageio.imwrite("output/supported_waves_training.mp4", wave_imgs)
    #model.prune_gaussians(1./500.)

    with torch.no_grad():
        spot = 0
        output = torch.empty_like(y)
        while spot < x.shape[0]:
            end_spot = min(spot+max_ls_points, x.shape[0])
            output[spot:end_spot] = model(x[spot:end_spot])
            spot = end_spot

        p = psnr(output,y).item()
        print(f"Final PSNR: {p:0.02f}")
    

        err = torch.clamp(((y-output)**2), 0., 1.)**0.5
        writer.add_scalar("Params vs. PSNR", 
                                      p, 
                                      model.param_count())
    #print(err.min())
    #print(err.max())
    plt.scatter(x.detach().cpu().numpy()[:,1],
                -x.detach().cpu().numpy()[:,0],
                c=err.detach().cpu().numpy().reshape(-1, model.n_channels))
    plt.savefig("./output/supportedperiodicprims_output.png")

    writer.flush()
    writer.close()
    
