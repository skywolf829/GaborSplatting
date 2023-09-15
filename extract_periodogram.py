import torch
import numpy as np
from models.LombScargle import MyLombScargleModel
from models.LombScargle2D import LombScargle2D
from models.Clustering import viz_PCA, viz_TSNE
import matplotlib.pyplot as plt
from utils.data_generators import load_img, sample_img_points
from tqdm import tqdm

if __name__ == "__main__":

    np.random.seed(0)
    torch.random.manual_seed(0)
    n_points = 100000

    device = "cuda"
    
    # Load data
    img = load_img("./data/tablecloth_zoom.jpg")
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

    with torch.no_grad():
        #new_points = torch.linspace(-2, 2, 1000)
        newx = torch.linspace(x[:,0].min(), x[:,0].max(), img.shape[0], device=device)
        newy = torch.linspace(x[:,1].min(), x[:,1].max(), img.shape[1], device=device)
        g = torch.stack(torch.meshgrid([newx, newy], indexing='ij'), dim=-1).reshape(-1, 2).type(torch.float32)
        
        new_img = np.zeros_like(img)

        ls_model = LombScargle2D(x,y-y.mean(dim=0), n_terms=1, device=device)
        freqs = torch.linspace(0.01, 20, 256)
        ls_model.fit(freqs)
        ls_model.find_peaks(top_n=1)
        ls_model.plot_power()
        
        peak_coeff = ls_model.get_peak_coeffs()
        peak_freqs = ls_model.get_peak_freqs()
        peak_power = ls_model.get_peak_power()
        color = ls_model.get_PCA_color()[:,None]+y.mean(dim=0).cuda()[:,None]
    print(f"Peak color { color }")
    print("original peaks")
    print(peak_coeff)
    xm = 2*torch.pi*peak_freqs[0,0]*g[:,0]
    ym = 2*torch.pi*peak_freqs[0,1]*g[:,1]
    channels = []
    for i in range(peak_coeff.shape[1]):
        chan = peak_coeff[0,i,0]*torch.sin(xm)*torch.sin(ym) + \
            peak_coeff[0,i,1]*torch.sin(xm)*torch.cos(ym) + \
            peak_coeff[0,i,2]*torch.sin(xm) + \
            peak_coeff[0,i,3]*torch.cos(xm)*torch.sin(ym) + \
            peak_coeff[0,i,4]*torch.cos(xm)*torch.cos(ym) + \
            peak_coeff[0,i,5]*torch.cos(xm) + \
            peak_coeff[0,i,6]*torch.sin(ym) + \
            peak_coeff[0,i,7]*torch.cos(ym) + \
            peak_coeff[0,i,8]
        channels.append(chan)
    rgb = torch.stack(channels, dim=-1) @ color.mT
    rgb = rgb.reshape(img.shape[0], img.shape[1], rgb.shape[-1])
    plt.imshow(rgb.cpu().numpy())
    plt.show()

    new_coeffs = ls_model.to_two_wave_form(peak_coeff)
    new_coeffs = ls_model.to_one_wave_form(new_coeffs)
    print("1 wave form peaks")
    print(new_coeffs)
    xm = 2*torch.pi*peak_freqs[0,0]*g[:,0]
    ym = 2*torch.pi*peak_freqs[0,1]*g[:,1]
    channels = []
    peak_color = new_coeffs[0,:,]
    for i in range(peak_coeff.shape[1]):
        chan = (new_coeffs[0,i,0]*torch.sin(xm+new_coeffs[0,i,1]) + \
            new_coeffs[0,i,2]) * \
            (new_coeffs[0,i,3]*torch.sin(ym+new_coeffs[0,i,4]) + \
            new_coeffs[0,i,5])
        channels.append(chan)
    rgb = torch.stack(channels, dim=-1) @ color.mT
    rgb = rgb.reshape(img.shape[0], img.shape[1], rgb.shape[-1])
    plt.imshow(rgb.cpu().numpy())
    plt.show()

        