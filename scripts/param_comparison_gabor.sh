#!/bin/sh
python train.py --training_data baboon.jpg --num_total_prims 10000 --save_name baboon_10000
python train.py --training_data baboon.jpg --num_total_prims 50000 --save_name baboon_50000
python train.py --training_data baboon.jpg --num_total_prims 80000 --save_name baboon_80000
python train.py --training_data baboon.jpg --num_total_prims 150000 --save_name baboon_150000
python train.py --training_data baboon.jpg --num_total_prims 300000 --save_name baboon_300000
python train.py --training_data baboon.jpg --num_total_prims 500000 --save_name baboon_500000

python train.py --training_data cameras.jpg --num_total_prims 100000 --save_name cameras_100000
python train.py --training_data cameras.jpg --num_total_prims 400000 --save_name cameras_400000
python train.py --training_data cameras.jpg --num_total_prims 700000 --save_name cameras_700000
python train.py --training_data cameras.jpg --num_total_prims 1000000 --save_name cameras_1000000
python train.py --training_data cameras.jpg --num_total_prims 1250000 --save_name cameras_1250000
python train.py --training_data cameras.jpg --num_total_prims 1500000 --save_name cameras_1500000

python train.py --training_data lighthouse.jpg --num_total_prims 100000 --save_name lighthouse_100000
python train.py --training_data lighthouse.jpg --num_total_prims 400000 --save_name lighthouse_400000
python train.py --training_data lighthouse.jpg --num_total_prims 700000 --save_name lighthouse_700000
python train.py --training_data lighthouse.jpg --num_total_prims 1000000 --save_name lighthouse_1000000
python train.py --training_data lighthouse.jpg --num_total_prims 1250000 --save_name lighthouse_1250000
python train.py --training_data lighthouse.jpg --num_total_prims 1500000 --save_name lighthouse_1500000

python train.py --training_data newyork.jpg --num_total_prims 100000 --save_name newyork_100000
python train.py --training_data newyork.jpg --num_total_prims 400000 --save_name newyork_400000
python train.py --training_data newyork.jpg --num_total_prims 700000 --save_name newyork_700000
python train.py --training_data newyork.jpg --num_total_prims 1000000 --save_name newyork_1000000
python train.py --training_data newyork.jpg --num_total_prims 1250000 --save_name newyork_1250000
python train.py --training_data newyork.jpg --num_total_prims 1500000 --save_name newyork_1500000


python train.py --training_data stainedglass.jpg --num_total_prims 100000 --save_name stainedglass_100000
python train.py --training_data stainedglass.jpg --num_total_prims 400000 --save_name stainedglass_400000
python train.py --training_data stainedglass.jpg --num_total_prims 700000 --save_name stainedglass_700000
python train.py --training_data stainedglass.jpg --num_total_prims 1000000 --save_name stainedglass_1000000
python train.py --training_data stainedglass.jpg --num_total_prims 1250000 --save_name stainedglass_1250000
python train.py --training_data stainedglass.jpg --num_total_prims 1500000 --save_name stainedglass_1500000


python train.py --training_data trees.jpeg --num_total_prims 100000 --save_name trees_100000
python train.py --training_data trees.jpeg --num_total_prims 400000 --save_name trees_400000
python train.py --training_data trees.jpeg --num_total_prims 700000 --save_name trees_700000
python train.py --training_data trees.jpeg --num_total_prims 1000000 --save_name trees_1000000
python train.py --training_data trees.jpeg --num_total_prims 1250000 --save_name trees_1250000
python train.py --training_data trees.jpeg --num_total_prims 1500000 --save_name trees_1500000

python train.py --training_data girlwithpearlearring.jpg --num_total_prims 100000 --save_name earring_100000
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 400000 --save_name earring_400000
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 700000 --save_name earring_700000
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1000000 --save_name earring_1000000
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1250000 --save_name earring_1250000
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1500000 --save_name earring_1500000


python train.py --training_data pluto.png --num_total_prims 100000 --save_name pluto_100000
python train.py --training_data pluto.png --num_total_prims 400000 --save_name pluto_400000
python train.py --training_data pluto.png --num_total_prims 700000 --save_name pluto_700000
python train.py --training_data pluto.png --num_total_prims 1000000 --save_name pluto_1000000
python train.py --training_data pluto.png --num_total_prims 1250000 --save_name pluto_1250000
python train.py --training_data pluto.png --num_total_prims 1500000 --save_name pluto_1500000


python train.py --training_data tokyo_gigapixel.jpg --num_total_prims 1000000 --save_name tokyo_1000000 --data_device cpu
python train.py --training_data tokyo_gigapixel.jpg --num_total_prims 2000000 --save_name tokyo_2000000 --data_device cpu
python train.py --training_data tokyo_gigapixel.jpg --num_total_prims 3000000 --save_name tokyo_3000000 --data_device cpu