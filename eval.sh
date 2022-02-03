#!/bin/bash

python evaluate.py --checkpoint='./output/1000/checkpoint_63.pth' --data_path='./test2.npz'
python urdf.py --checkpoint='./output/1000/checkpoint_63.pth' --save_dir='./2Visualize/test_urdf.xml'
