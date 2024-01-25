#!/bin/sh

python train.py --training_data girlwithpearlearring.jpg --num_total_prims 50000 --save_name earring_50000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 100000 --save_name earring_100000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 400000 --save_name earring_400000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 700000 --save_name earring_700000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1000000 --save_name earring_1000000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1250000 --save_name earring_1250000_4
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1350000 --save_name earring_1350000_4