#!/bin/bash

python evaluate.py --checkpoint='./output/yb/checkpoint_79.pth' --data_path='./test.npz'
python urdf.py --checkpoint='./output/yb/checkpoint_79.pth' --save_dir='./2Visualize/test_urdf.xml'
