import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio

gaussian_splatting_results = {
    "pluto": {
        "PSNR": [
            [10000*8, 24.86],
            [100000*8, 28.79],
            [500000*8, 31.16],
            [1000000*8, 32.45],
        ],
        "SSIM": [
            [10000*8, 0.7671],
            [100000*8, 0.8156],
            [500000*8, 0.8200],
            [1000000*8, 0.8681],
        ],
        "LPIPS": [
            [10000*8, 0.3414],
            [100000*8, 0.3000],
            [500000*8, 0.2374],
            [1000000*8, 0.1963],
        ]
    },
    "earring": {
        "PSNR": [
            [10000*8, 22.59],
            [100000*8, 23.19],
            [500000*8, 23.68],
            [700000*8, 23.87],
            [1000000*8, 24.08],
        ],
        "SSIM": [
            [10000*8, 0.5235],
            [100000*8, 0.5370],
            [500000*8, 0.5545],
            [700000*8, 0.5618],
            [1000000*8, 0.5716],
        ],
        "LPIPS": [
            [10000*8, 0.6377],
            [100000*8, 0.6510],
            [500000*8, 0.6331],
            [700000*8, 0.6249],  
            [1000000*8, 0.6140],          
        ]
    },
    "baboon": {
        "PSNR": [
            [10000*8, 21.52], 
            [50000*8, 30.94], 
            [100000*8, 35.68], 
            [500000*8, 54.84], 
            [1000000*8, 53.00], 
        ],
        "SSIM": [
            [10000*8, 0.4999], 
            [50000*8, 0.8628], 
            [100000*8, 0.9281], 
            [500000*8, 0.9998], 
            [1000000*8, 0.9999],
        ],
        "LPIPS": [
            [10000*8, 0.4492],             
            [50000*8, 0.1146], 
            [100000*8, 0.0600], 
            [500000*8, 0.0000], 
            [1000000*8, 0.0001],
        ]
    },
    "gigapixel": {
        "PSNR": [
            [100000*8, 23.09], 
            [300000*8, 24.35],
            [500000*8, 24.84], 
            [700000*8, 25.12], 
            [1000000*8, 25.39], 
        ]
    }
}

periodic_primitives_results = {
    "pluto": {
        "PSNR": [
            [10000*12, 26.09],
            [100000*12, 30.41],
            [500000*12, 33.65],
            [700000*12, 34.38],
        ],
        "SSIM": [
            [10000*12, 0.7083],
            [100000*12, 0.8218],
            [500000*12, 0.8869],
            [700000*12, 0.8996],
        ],
        "LPIPS": [
            [10000*12, 0.4033],
            [100000*12, 0.2465],
            [500000*12, 0.1408],
            [700000*12, 0.1182],
        ]
    },
    "earring": {
        "PSNR": [
            [10000*12, 22.93],
            [100000*12, 24.00],
            [500000*12, 25.34], 
            [700000*12, 25.64],
        ],
        "SSIM": [
            [10000*12, 0.5270],
            [100000*12, 0.5674],
            [500000*12, 0.6176], 
            [700000*12, 0.6297],
        ],
        "LPIPS": [    
            [10000*12, 0.6673],
            [100000*12, 0.5852], 
            [500000*12, 0.5352],        
            [700000*12, 0.5163],
        ]
    },
    "baboon": {
        "PSNR": [
            [10000*12, 27.32], 
            [50000*12, 42.49], 
            [100000*12, 57.00], 
            [500000*12, 63.53], 
        ],
        "SSIM": [
            [10000*12, 0.8579], 
            [50000*12, 0.9851], 
            [100000*12, 0.9996], 
            [500000*12, 1.0000], 
        ],
        "LPIPS": [
            [10000*12, 0.0555], 
            [50000*12, 0.0027], 
            [100000*12, 0.0000], 
            [500000*12, 0.0000], 
        ]
    },
    "gigapixel": {
        "PSNR": [
            [50000*12, 23.05], 
            [100000*12, 23.95], 
            [300000*12, 25.35], 
            [500000*12, 25.95], 
            [700000*12, 26.32], 
        ]
    }
}

def generate_charts():
    
    num_datasets = len(gaussian_splatting_results.keys())
    dataset_no = 0
    for dataset in gaussian_splatting_results.keys():
        plt.suptitle(dataset)
        num_metrics = len(gaussian_splatting_results[dataset].keys())
        metric_no = 0
        for metric in gaussian_splatting_results[dataset].keys():            
            plt.subplot(1, num_metrics, metric_no+1)
            plt.title(metric)            

            # Gaussian splatting
            color="orange"
            marker="o"
            results = np.array(gaussian_splatting_results[dataset][metric])
            plt.plot(results[:,0], results[:,1], color=color, marker=marker, label="Gaussians")       
            
            # Periodic primitives
            color="blue"
            marker="^"
            results = np.array(periodic_primitives_results[dataset][metric])
            plt.plot(results[:,0], results[:,1], color=color, marker=marker, label="Ours") 

            metric_no += 1
        plt.legend()
        plt.show()
        plt.clf()
        dataset_no += 1

def crop_imgs():
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = 933120000

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(project_folder_path, "data")
    output_folder = os.path.join(project_folder_path, "output")
    save_folder = os.path.join(project_folder_path, "savedModels")

    pluto_gt = imageio.imread(os.path.join(data_folder, "pluto.png"))
    earring_gt = imageio.imread(os.path.join(data_folder, "girlwithpearlearring.jpg"))
    baboon_gt = imageio.imread(os.path.join(data_folder, "baboon.jpg"))
    gigapixel_gt = imageio.imread(os.path.join(data_folder, "gigapixel.jpg"))

    pluto_gaussians = imageio.imread(os.path.join(output_folder, "pluto_1000000_gaussians.jpg"))
    earring_gaussians = imageio.imread(os.path.join(output_folder, "earring_1000000_gaussians.jpg"))
    baboon_gaussians = imageio.imread(os.path.join(output_folder, "baboon_100000_gaussians.jpg"))
    gigapixel_gaussians = imageio.imread(os.path.join(output_folder, "gigapixel_1000000_gaussians.jpg"))

    pluto_ours = imageio.imread(os.path.join(output_folder, "pluto_700000.jpg"))
    earring_ours = imageio.imread(os.path.join(output_folder, "earring_700000.jpg"))
    baboon_ours = imageio.imread(os.path.join(output_folder, "baboon_50000.jpg"))
    gigapixel_ours = imageio.imread(os.path.join(output_folder, "gigapixel_700000.jpg"))

    if(not os.path.exists(os.path.join(output_folder, "crops", "pluto"))):
        os.makedirs(os.path.join(output_folder, "crops", "pluto"))
    if(not os.path.exists(os.path.join(output_folder, "crops", "earring"))):
        os.makedirs(os.path.join(output_folder, "crops", "earring"))
    if(not os.path.exists(os.path.join(output_folder, "crops", "baboon"))):
        os.makedirs(os.path.join(output_folder, "crops", "baboon"))
    if(not os.path.exists(os.path.join(output_folder, "crops", "gigapixel"))):
        os.makedirs(os.path.join(output_folder, "crops", "gigapixel"))

    pluto_gt_c1 = pluto_gt[2145:2800, 5500:6000,:]
    pluto_gaussians_c1 = pluto_gaussians[2145:2800, 5500:6000,:]
    pluto_ours_c1 = pluto_ours[2145:2800, 5500:6000,:]

    pluto_gt_c2 = pluto_gt[4800:5300, 5500:6000,:]
    pluto_gaussians_c2 = pluto_gaussians[4800:5300, 5500:6000,:]
    pluto_ours_c2 = pluto_ours[4800:5300, 5500:6000,:]

    pluto_gt_c3 = pluto_gt[3800:4200, 2000:2500,:]
    pluto_gaussians_c3 = pluto_gaussians[3800:4200, 2000:2500,:]
    pluto_ours_c3 = pluto_ours[3800:4200, 2000:2500,:]

    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gt_c1.png"), pluto_gt_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gaussians_c1.png"), pluto_gaussians_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_ours_c1.png"), pluto_ours_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gt_c2.png"), pluto_gt_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gaussians_c2.png"), pluto_gaussians_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_ours_c2.png"), pluto_ours_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gt_c3.png"), pluto_gt_c3)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_gaussians_c3.png"), pluto_gaussians_c3)
    imageio.imwrite(os.path.join(output_folder, "crops", "pluto", "pluto_ours_c3.png"), pluto_ours_c3)

    
    earring_gt_c1 = earring_gt[2145:2800, 5500:6000,:]
    earring_gaussians_c1 = earring_gaussians[2145:2800, 5500:6000,:]
    earring_ours_c1 = earring_ours[2145:2800, 5500:6000,:]

    earring_gt_c2 = earring_gt[7000:7500, 4500:5000,:]
    earring_gaussians_c2 = earring_gaussians[7000:7500, 4500:5000,:]
    earring_ours_c2 = earring_ours[7000:7500, 4500:5000,:]

    earring_gt_c3 = earring_gt[3000:4000, 3000:4000,:]
    earring_gaussians_c3 = earring_gaussians[3000:4000, 3000:4000,:]
    earring_ours_c3 = earring_ours[3000:4000, 3000:4000,:]

    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gt_c1.png"), earring_gt_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gaussians_c1.png"), earring_gaussians_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_ours_c1.png"), earring_ours_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gt_c2.png"), earring_gt_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gaussians_c2.png"), earring_gaussians_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_ours_c2.png"), earring_ours_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gt_c3.png"), earring_gt_c3)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_gaussians_c3.png"), earring_gaussians_c3)
    imageio.imwrite(os.path.join(output_folder, "crops", "earring", "earring_ours_c3.png"), earring_ours_c3)

    baboon_gt_c1 = baboon_gt[256:512, 0:128,:]
    baboon_gaussians_c1 = baboon_gaussians[256:512, 0:128,:]
    baboon_ours_c1 = baboon_ours[256:512, 0:128,:]

    baboon_gt_c2 = baboon_gt[0:128, 256:440,:]
    baboon_gaussians_c2 = baboon_gaussians[0:128, 256:440,:]
    baboon_ours_c2 = baboon_ours[0:128, 256:440,:]


    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_gt_c1.png"), baboon_gt_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_gaussians_c1.png"), baboon_gaussians_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_ours_c1.png"), baboon_ours_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_gt_c2.png"), baboon_gt_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_gaussians_c2.png"), baboon_gaussians_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "baboon", "baboon_ours_c2.png"), baboon_ours_c2)

    gigapixel_gt_c1 = gigapixel_gt[15000:19000, 25000:29000,:]
    gigapixel_gaussians_c1 = gigapixel_gaussians[15000:19000, 25000:29000,:]
    gigapixel_ours_c1 = gigapixel_ours[15000:19000, 25000:29000,:]

    gigapixel_gt_c2 = gigapixel_gt[4000:8000, 34000:38000,:]
    gigapixel_gaussians_c2 = gigapixel_gaussians[4000:8000, 34000:38000,:]
    gigapixel_ours_c2 = gigapixel_ours[4000:8000, 34000:38000,:]


    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_gt_c1.png"), gigapixel_gt_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_gaussians_c1.png"), gigapixel_gaussians_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_ours_c1.png"), gigapixel_ours_c1)
    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_gt_c2.png"), gigapixel_gt_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_gaussians_c2.png"), gigapixel_gaussians_c2)
    imageio.imwrite(os.path.join(output_folder, "crops", "gigapixel", "gigapixel_ours_c2.png"), gigapixel_ours_c2)


#generate_charts()
crop_imgs()