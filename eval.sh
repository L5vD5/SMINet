#!/bin/bash

python evaluate.py --checkpoint='./output/0119_1/checkpoint_20.pth' --data_path='./data/Multi_2dim_log_spiral/fold9/Multi_2dim_log_spiral_910.txt' 
python urdf.py --checkpoint='./output/0119_1/checkpoint_20.pth' --save_dir='./2Visualize'
