import torch
import numpy as np
from models.LombScargle import MyLombScargleModel
from models.Clustering import viz_PCA, viz_TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":

    np.random.seed(0)
    torch.random.manual_seed(0)
    n_points = 100000
    
    # Load data
    #img = load_img("./data/tablecloth_zoom.jpg")
    #x,y = sample_img_points(img, n_points, plot=True)
    x = np.load("./data/point_positions.npy")
    y = np.load("./data/point_colors.npy")
    all_attributes = np.load('./data/point_attributes.npy')
     
    #x, y = generate_1D_random_peroidic_data_square(resolution)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    print(f"Data: x->f(x) {x.shape} -> {y.shape}")
    #plt.scatter(x[:,1], x[:,0], c=y, cmap='gray', s=0.1)
    #plt.show()
    viz_PCA(all_attributes,color=y)
    viz_TSNE(all_attributes,color=y)

    # Create model
    with torch.no_grad():
        ls_model = MyLombScargleModel(x,y, device="cuda")
        dist = (x.max(dim=0).values-x.min(dim=0).values).max()
        frequencies = torch.flip(1/torch.linspace(dist/4096, dist/2, 1024), dims=[0])
        angles = None
        if(x.shape[1] > 1):
            angles = [torch.linspace(0, torch.pi, 180)]

        ls_model.fit(frequencies, angles, perfect_fit=False)
        ls_model.find_peaks()
        ls_model.plot_power()
        
        #new_points = torch.linspace(-2, 2, 1000)
        newx = torch.linspace(x[:,0].min(), x[:,0].max(), 1024)
        newy = torch.linspace(x[:,1].min(), x[:,1].max(), 1024)
        g = torch.stack(torch.meshgrid([newx, newy], indexing='ij'), dim=-1).reshape(-1, 2).type(torch.float32)
        
        
        psnrs = []
        ssims = []
        num_peaks = []
        for n_peaks in np.linspace(1, ls_model.get_num_peaks(), 10):
            new_points = ls_model.transform_from_peaks(g, top_n_peaks=int(n_peaks)).reshape(1024, 1024).cpu().numpy()  
            plt.imshow(new_points, cmap='gray')
            plt.show()         
            #ssims.append(ssim(img, new_points))
            #new_points -= img
            #new_points **= 2
            #p = -10*np.log10(new_points.mean())
            #psnrs.append(p)
            #num_peaks.append(n_peaks)
        quit() 
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()

        ax1.plot(num_peaks, psnrs)
        ax1.set_ylabel("PSNR (dB)")
        ax2.plot(num_peaks, ssims, linestyle='dashed')
        ax2.set_ylabel("SSIM")
        ax1.set_xlabel("Num peaks")
        
        ax1.set_title("Quality vs num. peaks")
        plt.show()
    
        