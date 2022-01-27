#!/bin/bash

python train.py --batch_size=128 --data_path='./output.npz' --save_dir='./output/yb' --device='1' --epochs=5000 --wandb --Foldstart=0 --Foldend=8 --input_dim=1
