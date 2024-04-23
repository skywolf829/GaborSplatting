#!/bin/sh
python train.py --training_data Aalesund.jpg                    --num_total_prims 3000000 --save_name Aalesund_Gabor --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data DenmanProspect.jpg              --num_total_prims 3000000 --save_name DenmanProspect_Gabor --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data MtCook.jpg                      --num_total_prims 3000000 --save_name MtCook_Gabor --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data Norway.jpg                      --num_total_prims 3000000 --save_name Norway_Gabor --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data Vermont.jpg                     --num_total_prims 3000000 --save_name Vermont_Gabor --data_device cpu  --log_every 0 --log_image_every 0
python train.py --training_data tokyo_gigapixel.jpg             --num_total_prims 3000000 --save_name Tokyo_Gabor --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data pluto.png                       --num_total_prims 3000000 --save_name Pluto_Gabor --log_every 0 --log_image_every 0
python train.py --training_data lighthouse.jpg                  --num_total_prims 3000000 --save_name Lighthouse_Gabor --log_every 0 --log_image_every 0
python train.py --training_data girlwithpearlearring_giga.jpg   --num_total_prims 3000000 --save_name GirlWithPearlEarring_Gabor --log_every 0 --log_image_every 0
python train.py --training_data TapestryofBlazingStarbirth.jpg  --num_total_prims 3000000 --save_name Starbirth_Gabor --log_every 0 --log_image_every 0

python train.py --training_data Aalesund.jpg                    --num_total_prims 3750000 --save_name Aalesund_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data DenmanProspect.jpg              --num_total_prims 3750000 --save_name DenmanProspect_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data MtCook.jpg                      --num_total_prims 3750000 --save_name MtCook_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data Norway.jpg                      --num_total_prims 3750000 --save_name Norway_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0 
python train.py --training_data Vermont.jpg                     --num_total_prims 3750000 --save_name Vermont_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data tokyo_gigapixel.jpg             --num_total_prims 3750000 --save_name Tokyo_Gaussian --data_device cpu --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data pluto.png                       --num_total_prims 3750000 --save_name Pluto_Gaussian --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data lighthouse.jpg                  --num_total_prims 3750000 --save_name Lighthouse_Gaussian --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data girlwithpearlearring_giga.jpg   --num_total_prims 3750000 --save_name GirlWithPearlEarring_Gaussian --gaussian_only 1 --log_every 0 --log_image_every 0
python train.py --training_data TapestryofBlazingStarbirth.jpg  --num_total_prims 3750000 --save_name Starbirth_Gaussian --gaussian_only 1 --log_every 0 --log_image_every 0

python train.py --training_data Aalesund.jpg                    --model iNGP --save_name Aalesund_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data DenmanProspect.jpg              --model iNGP --save_name DenmanProspect_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data MtCook.jpg                      --model iNGP --save_name MtCook_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data Norway.jpg                      --model iNGP --save_name Norway_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data Vermont.jpg                     --model iNGP --save_name Vermont_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data tokyo_gigapixel.jpg             --model iNGP --save_name Tokyo_iNGP --data_device cpu --log_every 0 --log_image_every 0
python train.py --training_data pluto.png                       --model iNGP --save_name Pluto_iNGP --log_every 0 --log_image_every 0
python train.py --training_data lighthouse.jpg                  --model iNGP --save_name Lighthouse_iNGP --log_every 0 --log_image_every 0
python train.py --training_data girlwithpearlearring_giga.jpg   --model iNGP --save_name GirlWithPearlEarring_iNGP --log_every 0 --log_image_every 0
python train.py --training_data TapestryofBlazingStarbirth.jpg  --model iNGP --save_name Starbirth_iNGP --log_every 0 --log_image_every 0
