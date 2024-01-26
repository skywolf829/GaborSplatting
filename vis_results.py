import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v3 as imageio
from models.options import load_options
import json
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

project_folder_path = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(project_folder_path, "output")
save_folder = os.path.join(project_folder_path, "savedModels")

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

def plot_chart(dataset_name, dataset_results):
    
    num_metrics = 3

    with plt.style.context("seaborn-paper"):
        fig, axs = plt.subplots(1, num_metrics, figsize=(4*3, 4))
        fig.suptitle(dataset_name)
        num_options = len(dataset_results.keys())
        formatter = FuncFormatter(millions)
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', "pink"]
        markers = ['o','v','x','s','^',"."]

        if(num_options > min(len(colors), len(markers))):
            print(f"Need more color/marker choices for the options you have!")
            return

        option_no = 0
        for option in dataset_results.keys():           
            num_params = np.array(dataset_results[option]["num_params"])
            sorted_order = np.argsort(num_params)
            sorted_params = num_params[sorted_order]

            metric_no = 0
            for metric in dataset_results[option].keys():
                if metric == "num_params":
                    continue
                ax = axs[metric_no]
                ax.xaxis.set_major_formatter(millions)   
                ax.set_title(metric)
                ax.set_xlabel("Num params")

                # Periodic primitives
                color=colors[option_no]
                marker=markers[option_no]
                sorted_metric = np.array(dataset_results[option][metric])[sorted_order]
                ax.plot(sorted_params, sorted_metric, color=color, marker=marker, label=option) 
                

                metric_no += 1
            option_no += 1

        axs[-1].legend()
        fig.show()
        plt.waitforbuttonpress()

def generate_charts():
    # an object to hold all results, broken up by dataset type
    results = {}

    # iterate through all saved models
    for savedModel in os.listdir(save_folder):
        
        # check if the model, options file, and results file exists
        checks = os.path.exists(os.path.join(save_folder, savedModel, "model.ckpt.npz")) and \
                os.path.exists(os.path.join(save_folder, savedModel, "options.json")) and \
                os.path.exists(os.path.join(save_folder, savedModel, "results.json"))

        # skip this model if it is missing some part 
        if not checks:
            continue

        # get the model details from the options file
        opt = load_options(os.path.join(save_folder, savedModel))
        # load the model's results
        fp = open(os.path.join(save_folder, savedModel, "results.json"))
        result = json.load(fp)
        fp.close()
        
        # categorize results by image and by the training setup
        image_name = opt['training_data'].split(".")[0]
        suboptions = "Gaussian" if opt['gaussian_only'] else f"Gabor"
        
        # for frequency only tests
        #if opt['gaussian_only']:
        #    continue
        # for ours (2 frequencies) vs GS only
        #if opt['num_frequencies'] != 2 and not opt['gaussian_only']:
        #    continue
        
        if image_name not in results.keys():
            results[image_name] = {}
        if suboptions not in results[image_name].keys():
            results[image_name][suboptions] = {}
        
        for metric in result.keys():
            if "PSNR" in metric or "SSIM" in metric or "LPIPS" in metric or "num_params" in metric:
                if metric not in results[image_name][suboptions].keys():
                    results[image_name][suboptions][metric] = []
                results[image_name][suboptions][metric].append(result[metric])

    # next, make a chart for each image
    
    for image_name in results.keys():
        plot_chart(image_name, results[image_name])

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

def avg_metrics_F_test():
    fold = os.path.join(save_folder, "F_test")
    metrics = {}
    for model in os.listdir(fold):
        if(".txt" in model):
            continue
        k = int(model.split("_")[-1])
        if k not in metrics.keys():
            metrics[k] = {"n": 0}
        fp = open(os.path.join(fold, model, "results.json"))
        result = json.load(fp)
        fp.close()

        for key in result.keys():
            if key not in metrics[k].keys():
                metrics[k][key] = 0
            metrics[k][key] += float(result[key])
        metrics[k]['n'] += 1
    
    for k in metrics.keys():
        print(f"F={k}")
        n = metrics[k]["n"]
        for key in metrics[k]:
            print(f"Average {key}: {metrics[k][key]/n : 0.03f}")

        print()

def avg_metrics_k_test():
    fold = os.path.join(save_folder, "k_test")
    metrics = {}
    for model in os.listdir(fold):
        if(".txt" in model):
            continue
        k = int(model.split("_")[-1])
        if k not in metrics.keys():
            metrics[k] = {"n": 0}
        fp = open(os.path.join(fold, model, "results.json"))
        result = json.load(fp)
        fp.close()

        for key in result.keys():
            if key not in metrics[k].keys():
                metrics[k][key] = 0
            metrics[k][key] += float(result[key])
        metrics[k]['n'] += 1
    
    for k in metrics.keys():
        print(f"k={k}")
        n = metrics[k]["n"]
        for key in metrics[k]:
            print(f"Average {key}: {metrics[k][key]/n : 0.03f}")

        print()

def avg_metrics_max_freq_test():
    fold = os.path.join(save_folder, "max_frequency_test")
    metrics = {}
    for model in os.listdir(fold):
        if(".txt" in model):
            continue
        k = int(model.split("_")[-1])
        if k not in metrics.keys():
            metrics[k] = {"n": 0}
        fp = open(os.path.join(fold, model, "results.json"))
        result = json.load(fp)
        fp.close()

        for key in result.keys():
            if key not in metrics[k].keys():
                metrics[k][key] = 0
            metrics[k][key] += float(result[key])
        metrics[k]['n'] += 1
    
    for k in metrics.keys():
        print(f"freq={k}")
        n = metrics[k]["n"]
        for key in metrics[k]:
            print(f"Average {key}: {metrics[k][key]/n : 0.03f}")

        print()

def create_gabor():
    x = np.linspace(-1.0, 1.0, 512)[None,:].repeat(512, 0)
    y = np.linspace(-1.0, 1.0, 512)[:,None].repeat(512, 1)
    xy = np.stack([x,y], axis=-1)
    print(xy.shape)
    g = np.exp(-(xy[:,:,0]*xy[:,:,0] + xy[:,:,1]*xy[:,:,1])/0.1)
    s = np.sin(16*(xy[:,:,0]*np.cos(2.1)+xy[:,:,1]*np.sin(2.1)))
    gab = g * s
    plt.imshow(gab, cmap="bwr")
    plt.show()
    

#generate_charts()
avg_metrics_F_test()    
#avg_metrics_k_test()
#avg_metrics_max_freq_test()    
#create_gabor()