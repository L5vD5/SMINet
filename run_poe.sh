#!/bin/bash

python train.py --lr=0.001 --batch_size=128 --Vec_loss=0 --q_entropy=0 --data_path='./data2.npz' --save_dir='./output/1000' --resume_dir='./output/1000/checkpoint_133.pth' --device='1' --epochs=500 --wandb --Foldstart=0 --Foldend=8 --input_dim=3
