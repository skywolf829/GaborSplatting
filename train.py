from models.GaussianSplatting2D import GaussianSplatting2D
from models.PeriodicPrimitives2D import PeriodicPrimitives2D
from models.HybridPrimitiveModel import HybridPrimitiveModel
from models.PeriodicGaussians2D import PeriodicGaussians2D
from models.PeriodicGaussianField import PeriodicGaussianField
from models.SupportedPeriodicPrimitives2D import SupportedPeriodicPrimitives2D
from models.HaarPrimitives2D import HaarPrimitives2D
from utils.data_generators import load_img
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm
from utils.data_utils import to_img, psnr
import time

if __name__ == '__main__':

    device = "cuda"
    torch.random.manual_seed(42)
    np.random.seed(42)
    
    training_img = load_img("./data/synthetic5.jpg")
    training_img_copy = (training_img.copy() * 255).astype(np.uint8)
    og_img_shape = training_img.shape
    model = PeriodicGaussians2D(1, device=device, n_channels=3)

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
    print(f"Initializing Lomb-Scargle model on training data...")

    iters_per_wave = 1000
    waves_per_ls = 1
    total_iters = int(iters_per_wave*model.n_waves/waves_per_ls)
    
    num_params = []

    pre_fitting_imgs = []
    wave_imgs = []
    max_tries = 5
    tries = max_tries
    
    max_ls_points = 2**17
    pct_of_data = max_ls_points / x.shape[0]
    t = tqdm(range(total_iters))
    for i in t:
        mask = torch.rand(x.shape[0], device=x.device, dtype=torch.float32) < pct_of_data

        # image logging
        if i % 50 == 0 and i > 0:
            with torch.no_grad():
                res = [200, 200]
                xmin = x.min(dim=0).values
                xmax = x.max(dim=0).values
                g = [torch.linspace(xmin[i], xmax[i], res[i], device=model.device) for i in range(xmin.shape[0])]
                g = torch.stack(torch.meshgrid(g, indexing='ij'), dim=-1).flatten(0, -2)
                img = model(g).reshape(res+[model.n_channels])
                img = to_img(img)
                pre_fitting_imgs.append(img)
                wave_img = model.vis_each_wave(x)
                wave_imgs.append(wave_img)

        # adding waves
        if i % iters_per_wave == 0:
            with torch.no_grad():    
                residuals = y[mask]
                if(i>0):
                    residuals -= model(x[mask])
                #model.prune_gaussians(1./500.)
                n_extracted_peaks = model.add_next_wave(x[mask],
                                    residuals,
                                    n_waves = waves_per_ls,
                                    n_freqs = 180, 
                                    freq_decay=1.01**((i//iters_per_wave)*waves_per_ls), 
                                    min_influence=1./500.)
                if(n_extracted_peaks == 0 and tries == 0):
                    print(f" Stopping wave detection early, no peaks found in {max_tries} iterations.")
                    break
                elif(n_extracted_peaks == 0):
                    tries -= 1
                else:
                    tries = max_tries
        
        #if i % iters_per_wave == 0 and i > 0:
            #with torch.no_grad():
                #model.colors.zero_()

        # actual training step
        model.optimizer.zero_grad()
        loss, model_out = model.loss(x[mask], y[mask])
        if (loss.isnan()):
            print()
            print(f"Detected loss was NaN.")
            quit()
            
        loss.backward()
        #print(f"{self.subgaussian_width} {self.subgaussian_width.grad}")
        model.optimizer.step()
        with torch.no_grad():
            model.subgaussian_flat_top_power.clamp_(-1, 3)
            
        with torch.no_grad():
            model.subgaussian_rotation +=  0.005
        # logging
        with torch.no_grad():             
            t.set_description(f"[{i+1}/{total_iters}] loss: {loss.item():0.04f}")
    #print(p.key_averages().table(
    #    sort_by="self_cuda_time_total", row_limit=-1))
    
    print(model.colors)
    imageio.imwrite("output/supported_waves_training_err.mp4", pre_fitting_imgs)
    imageio.imwrite("output/supported_waves_training.mp4", wave_imgs)
    #model.prune_gaussians(1./500.)
    print(f"Number of extracted waves: {model.gaussian_means.shape[0]}")
    output = model(x)
    p = psnr(output,y).item()
    print(f"Final PSNR: {p:0.02f}")
    

    err = torch.clamp(((y-output)**2), 0., 1.)
    print(err.min())
    print(err.max())
    plt.scatter(x.detach().cpu().numpy()[:,1],
                -x.detach().cpu().numpy()[:,0],
                c=err.detach().cpu().numpy().reshape(-1, model.n_channels))
    plt.savefig("./output/supportedperiodicprims_output.png")

