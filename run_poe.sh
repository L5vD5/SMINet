#!/bin/bash

python train.py --batch_size=128 --data_path='./data.npz' --save_dir='./output/yb' --device='1' --epochs=500 --wandb --Foldstart=0 --Foldend=8 --input_dim=3
