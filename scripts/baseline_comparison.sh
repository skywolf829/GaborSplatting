#!/bin/sh
python train.py --training_data baboon.jpg                  --num_total_prims 3000000 --save_name baboon_Gabor
python train.py --training_data cameras.jpg                 --num_total_prims 3000000 --save_name cameras_Gabor
python train.py --training_data lighthouse.jpg              --num_total_prims 3000000 --save_name lighthouse_Gabor
python train.py --training_data newyork.jpg                 --num_total_prims 3000000 --save_name newyork_Gabor
python train.py --training_data stainedglass.jpg            --num_total_prims 3000000 --save_name stainedglass_Gabor
python train.py --training_data trees.jpeg                  --num_total_prims 3000000 --save_name trees_Gabor
python train.py --training_data girlwithpearlearring.jpg    --num_total_prims 3000000 --save_name earring_Gabor
python train.py --training_data pluto.png                   --num_total_prims 3000000 --save_name pluto_Gabor
python train.py --training_data tokyo_gigapixel.jpg         --num_total_prims 3000000 --save_name tokyo_Gabor --data_device cpu

python train.py --training_data baboon.jpg                  --num_total_prims 3750000 --save_name baboon_Gaussian --gaussian_only 1
python train.py --training_data cameras.jpg                 --num_total_prims 3750000 --save_name cameras_Gaussian --gaussian_only 1
python train.py --training_data lighthouse.jpg              --num_total_prims 3750000 --save_name lighthouse_Gaussian --gaussian_only 1
python train.py --training_data newyork.jpg                 --num_total_prims 3750000 --save_name newyork_Gaussian --gaussian_only 1
python train.py --training_data stainedglass.jpg            --num_total_prims 3750000 --save_name stainedglass_Gaussian --gaussian_only 1
python train.py --training_data trees.jpeg                  --num_total_prims 3750000 --save_name trees_Gaussian --gaussian_only 1
python train.py --training_data girlwithpearlearring.jpg    --num_total_prims 3750000 --save_name earring_Gaussian --gaussian_only 1
python train.py --training_data pluto.png                   --num_total_prims 3750000 --save_name pluto_Gaussian --gaussian_only 1
python train.py --training_data tokyo_gigapixel.jpg         --num_total_prims 3750000 --save_name tokyo_Gaussian --data_device cpu --gaussian_only 1