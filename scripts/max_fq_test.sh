#!/bin/sh

python train.py --training_data girlwithpearlearring.jpg --num_total_prims 50000 --save_name earring_50000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 100000 --save_name earring_100000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 400000 --save_name earring_400000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 700000 --save_name earring_700000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1000000 --save_name earring_1000000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1250000 --save_name earring_1250000_128Hz
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1350000 --save_name earring_1350000_128Hz

python train.py --training_data girlwithpearlearring.jpg --num_total_prims 50000 --save_name earring_50000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 100000 --save_name earring_100000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 400000 --save_name earring_400000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 700000 --save_name earring_700000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1000000 --save_name earring_1000000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1250000 --save_name earring_1250000_64Hz --max_frequency 64
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1350000 --save_name earring_1350000_64Hz --max_frequency 64

python train.py --training_data girlwithpearlearring.jpg --num_total_prims 50000 --save_name earring_50000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 100000 --save_name earring_100000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 400000 --save_name earring_400000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 700000 --save_name earring_700000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1000000 --save_name earring_1000000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1250000 --save_name earring_1250000_32Hz --max_frequency 32
python train.py --training_data girlwithpearlearring.jpg --num_total_prims 1350000 --save_name earring_1350000_32Hz --max_frequency 32