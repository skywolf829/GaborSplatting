import torch
import numpy as np
from models.LombScargle import MyLombScargleModel
from models.LombScargle2D import LombScargle2D
from models.LombScargle2Danglefreq import LombScargle2Danglefreq
from models.Clustering import viz_PCA, viz_TSNE
import matplotlib.pyplot as plt
from utils.data_generators import load_img, sample_img_points
from tqdm import tqdm

if __name__ == "__main__":

    np.random.seed(0)
    torch.random.manual_seed(0)
    n_points = 10000

    device = "cuda"
    
    # Load data
    img = load_img("./data/synthetic10.jpg")
    x,y = sample_img_points(img, n_points, plot=False)
    #x = np.load("./data/point_positions.npy")
    #y = np.load("./data/point_colors.npy")
    #all_attributes = np.load('./data/point_attributes.npy')
     
    #x, y = generate_1D_random_peroidic_data_square(resolution)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    print(f"Data: x->f(x) {x.shape} -> {y.shape}. [{x.min()},{x.max()}] -> [{y.min()},{y.max()}]")
    #plt.scatter(x[:,1], x[:,0], c=y, cmap='gray', s=0.1)
    #plt.show()
    #viz_PCA(all_attributes,color=y)
    #viz_TSNE(all_attributes,color=y)
    print(f"avg color {y.mean(dim=0)}")
    with torch.no_grad():
        #new_points = torch.linspace(-2, 2, 1000)
        newx = torch.linspace(x[:,0].min(), x[:,0].max(), img.shape[0], device=device)
        newy = torch.linspace(x[:,1].min(), x[:,1].max(), img.shape[1], device=device)
        g = torch.stack(torch.meshgrid([newx, newy], indexing='ij'), dim=-1).reshape(-1, 2).type(torch.float32)
        
        new_img = np.zeros_like(img)

        ls_model = LombScargle2Danglefreq(x,y, n_terms=1, device=device)
        freqs = torch.linspace(0.2,48, 128)
        angles = torch.linspace(0, 0.5*torch.pi, 180)
        ls_model.fit(freqs, angles)
        n_extracted = ls_model.find_peaks(top_n=4)
        ls_model.plot_power()
        means, vars = ls_model.get_peak_placement(torch.arange(0, n_extracted, 1, dtype=torch.long, device=device))
        print(f"Wave positions {means} {vars}")
        peak_coeff = ls_model.get_peak_coeffs()
        peak_freqs = ls_model.get_peak_freqs()
        peak_power = ls_model.get_peak_power()
        color = ls_model.get_PCA_color()[:,None]
    print(f"Peak color { color }")
    print("original peaks")
    


        