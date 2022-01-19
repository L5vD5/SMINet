import torch
from model import *
from dataloader import *
from utils.pyart import *
import argparse
import numpy as np
from pathlib import Path

def main(args):
    print("Processing...")

    # make save_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    branchNum = checkpoint['branchNum']
    input_dim = checkpoint['input_dim']
    branchLs = bnum2ls(branchNum)
    n_joint = len(branchLs)

    # load model
    model = Model(branchLs, input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # load data
    test_data_loader = ToyDataloader(args.data_path, n_workers = 1, batch = 1, shuffle=False)

    # get jointAngle.txt
    revAngle = np.array([]).reshape(-1,n_joint)
    priAngle = np.array([]).reshape(-1,n_joint)
    for input,_ in test_data_loader:
        
        rev_q_value, pri_q_value = model.q_layer(input)

        rev_q_value = rev_q_value.detach().cpu().numpy()
        pri_q_value = pri_q_value.detach().cpu().numpy()

        revAngle = np.vstack((revAngle,rev_q_value))
        priAngle = np.vstack((priAngle,pri_q_value))
    np.savetxt(args.save_dir+"/revAngle.txt", revAngle)
    np.savetxt(args.save_dir+"/priAngle.txt", priAngle)

    # get branchLs
    np.savetxt(args.save_dir+'/branchLs.txt',branchLs)

    # get targetPose.txt
    targetPose = test_data_loader.dataset.label
    targetPose = targetPose.detach().cpu().numpy()
    np.savetxt(args.save_dir+'/targetPose.txt', targetPose)

    # get outputPose.txt
    outputPose = np.array([]).reshape(-1,targetPose.shape[1])
    for input,_ in test_data_loader:
        outputPose_temp,_ = model(input)
        outputPose_temp = outputPose_temp[:,:,0:3,3]
        outputPose_temp = outputPose_temp.reshape(-1,outputPose_temp.size()[1]*outputPose_temp.size()[2])
        outputPose_temp = outputPose_temp.detach().cpu().numpy()[0]
        outputPose = np.vstack((outputPose,outputPose_temp))
        
    np.savetxt(args.save_dir+"/outputPose.txt", outputPose)

    print("Done...")
if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--data_path', \
        default= './data/Multi_2dim_log_spiral/fold9/Multi_2dim_log_spiral_910.txt',type=str, \
            help='path to model checkpoint')    
    args.add_argument('--checkpoint', default= './output/0118/checkpoint_20.pth',type=str,
                    help='path to model checkpoint')
    args.add_argument('--save_dir', default='./2Visualize')
    args = args.parse_args()
    main(args)