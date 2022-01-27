#!/bin/bash

python evaluate.py --checkpoint='./output/yb/checkpoint_4918.pth' --data_path='./output.npz'
python urdf.py --checkpoint='./output/yb/checkpoint_4918.pth' --save_dir='./2Visualize/test_urdf.xml'
