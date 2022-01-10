#%%
import argparse
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *
from xml.dom import minidom 
from model import Model
#%%
def get_free_joint(PRset,joint,root,robot):
    revjointname = 'R'+str(joint+1)
    prijointname = 'P'+str(joint+1)

    ### For revolute joint

    # creat root element
    revjoint = root.createElement('joint') 
    revjoint.setAttribute('name',revjointname)
    revjoint.setAttribute('type','revolute')
    robot.appendChild(revjoint) 

    # set parent
    parent = root.createElement('parent')
    parentname = 'link_P'+str(joint)
    parent.setAttribute('link', parentname)
    revjoint.appendChild(parent)

    # set child
    child = root.createElement('child')
    childname = 'link_R'+str(joint+1)
    child.setAttribute('link',childname)
    revjoint.appendChild(child)

    # set origin for joint
    origin = root.createElement('origin')
    xyz = PRset.p_offset.unsqueeze(0).detach().cpu().numpy()
    rpy = PRset.rpy_offset.unsqueeze(0).detach().cpu().numpy()

    origin.setAttribute('xyz',xyz )
    origin.setAttribute('rpy',rpy )
    
    revjoint.appendChild(origin) 

    # set axis for joint
    axis = root.createElement('axis')
    xyz = PRset.rev_axis
    
    axis.setAttribute('xyz',xyz)
    revjoint.appendChild(axis)

    return robot


def main(args):
    statedict = torch.load(args.data_path)
    weight = statedict['state_dict']
    input_dim = statedict['input_dim']
    branchNum = statedict['branchNum']
    branchLs = bnum2ls(args.branchNum)
    
    
    model = Model(branchLs, input_dim)
    
    root = minidom.Document() 
    robot = root.createElement('robot')
    robot.setAttribute('name','softrobot')
    root.appendChild(robot)

    # get joint
    n_joint = len(branchLs)
    for joint in range(n_joint):
        
        PRset = getattr(model.trans_layer,'joint_'+str(joint+1))
        Trackbool = PRset.Trackbool
        if Trackbool:
            free_joint,tracking_joint = get_tracking_joint(PRset,joint,root,robot)
            
        else:
            free_joint = get_free_joint(PRset,joint)
        

    




if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--data_path', default= './data/Multi_2dim_log_spiral',type=str,
                    help='path to data')

    args = args.parse_args()
    main(args)