# Gabor Splatting
 
This repository contains code for a 2D primitive-based scene representation for images.
We use a gabor filter, which is a gaussian envelope multiplied by a waveform, to model images with higher reconstruction accuracy compared to typical gaussians, which have become popular from the seminal paper 3DGS (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

# Why Gabors?
In many domains, periodic functions are useful for a compressed representation. 
For example, image and video compression CODECs JPEG2000 and h.264 both use the Fourier domain to efficiently represent the data it is compressing.
Gabors are a way to bridge the gap between efficient periodic representations and the impressive results by Gaussian splatting.
We see the combined efficiency of using waveforms with the efficiency of using gaussians in one succinct representation, achieving higher quality representations at the same parameter counts as Gaussian splatting.

# Requirements

Our requirements are Python 3.7 and CUDA 11 (we use CUDA 11.8 SDK with PyTorch compiled with CUDA 11.6).
This will not run on a device without a CUDA enabled graphics card.

# Installation

We use Anaconda for package management.
Please install our environment with:

```conda env create --file conda_env.yml```
and
```conda activate periodic_primitives```

This will automatically install our CUDA module ```PeriodicPrimitives``` as well.

# Training a model

Please see examples of training a model in the ```scripts``` folder. 
We use ```train.py``` with various command line arguments, each of which can be understood using ```python train.py -h```.
To see default values for each option, please visit ```models/options.py```.

The model will be saved to ```savedModels/{save_name}```, with its parameters, options JSON file, and a results JSON file that shows the final PSNR, SSIM, and LPIPS.
A final output image will also be saved to ```output/{save_name}.png```.

# Other

## Changing frequency bank size or k

There are two hyperparamters that require recompiling the CUDA module.
If you would like to change the frequency bank size (```--num_total_frequencies``` option) or the number of frequencies term controlling k (```--num_frequencies``` option), you also need to change that value in the CUDA code and recompile.
Open file ```CUDA_modules/PeriodicPrimitivesCUDA/periodic_primitives2DRGB_cuda_kernel.cu```, and adjust lines 14 and 15 accordingly.
These are defined like so in order to create a shared memory array for blocks while processing the forward pass. 

## 3D

We have done some prelimenary experiments in 3D, but do not have success yet in outperforming gaussian splatting. 
We are happy to discuss if you have any suggestions!