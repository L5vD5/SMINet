#%%
import argparse
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *
from model import Model
import xml.etree.cElementTree as ET
from xml.dom import minidom


def list2str(ls):
    ls = [str(ele) for ele in ls]
    output = ' '.join(ls)

    return output

def get_joint(PRset,joint,robot, Trackbool=False):
    ''' For revolute joint '''
    revjointname = 'R'+str(joint+1)

    # create joint element
    revjoint = ET.SubElement(robot, "joint", name=revjointname, type='revolute')
    
    # set parent
    parentname = 'link_P'+str(joint)
    ET.SubElement(revjoint, "parent", link=parentname)

    # set child & declare child link
    childname = 'link_R'+str(joint+1)
    ET.SubElement(revjoint, "child", link=childname)
    ET.SubElement(robot,'link',name=childname)

    # set origin for joint
    xyz = PRset.p_offset.squeeze(0).detach().cpu().tolist()
    xyz = list2str(xyz)
    rpy = PRset.rpy_offset.squeeze(0).detach().cpu().tolist()
    rpy = list2str(rpy)
    ET.SubElement(revjoint, "origin", rpy=rpy, xyz=xyz)
    
    # set axis for joint
    xyz = PRset.rev_axis.detach().cpu().tolist()
    xyz = list2str(xyz)
    ET.SubElement(revjoint,"axis",xyz=xyz)
    

    ''' For prismatic joint '''
    prijointname = 'P'+str(joint+1)

    # create joint element
    prijoint = ET.SubElement(robot, "joint", name=prijointname, type='prismatic')
    
    # set parent
    parentname = 'link_R'+str(joint+1)
    ET.SubElement(prijoint, "parent", link=parentname)

    # set child & declare child link
    childname = 'link_P'+str(joint+1)
    ET.SubElement(prijoint, "child", link=childname)
    ET.SubElement(robot,'link',name=childname)

    # set origin for joint
    xyz = '0 0 0'
    rpy = '0 0 0'
    ET.SubElement(prijoint, "origin", rpy=rpy, xyz=xyz)
    
    # set axis for joint
    xyz = PRset.rev_axis.detach().cpu().tolist()
    xyz = list2str(xyz)
    ET.SubElement(prijoint,"axis",xyz=xyz)

    if Trackbool:
        # create joint element
        Trackname = 'T' + str(joint+1)
        Trackjoint = ET.SubElement(robot, "joint", name=Trackname, type='fixed')
        
        # set parent
        parentname = 'link_P'+str(joint+1)
        ET.SubElement(Trackjoint,'parent',link=parentname)

        # set child & declare child link
        childname = 'link_T'+str(joint+1)
        ET.SubElement(Trackjoint,'child',link=childname)
        ET.SubElement(robot,'link',name=childname)

        # set origin for joint
        xyz = PRset.p_track.squeeze(0).detach().cpu().tolist()
        xyz = list2str(xyz)
        rpy = PRset.rpy_track.squeeze(0).detach().cpu().tolist()
        rpy = list2str(rpy)
        ET.SubElement(Trackjoint,"origin",rpy=rpy,xyz=xyz)
        
    return robot

def set_ground(robot):
    ET.SubElement(robot,'link', name='world')
    ET.SubElement(robot,'link', name='base_link')
    ET.SubElement(robot,'link', name='link_P0')

    world_joint = ET.SubElement(robot,'joint',name='world_fixed',type='fixed')
    ET.SubElement(world_joint,'origin',xyz='0 0 0',rpy='0 0 0')
    ET.SubElement(world_joint,'parent',link='world')
    ET.SubElement(world_joint,'child',link='base_link')

    root_joint = ET.SubElement(robot,'joint',name='root_base',type='fixed')
    ET.SubElement(root_joint,'origin',xyz='0 0 0',rpy='0 0 0')
    ET.SubElement(root_joint,'parent',link='base_link')
    ET.SubElement(root_joint,'child',link='link_P0')

    return robot


def main(args):
    statedict = torch.load(args.data_path)
    weight = statedict['state_dict']
    input_dim = statedict['input_dim']
    branchNum = statedict['branchNum']
    branchLs = bnum2ls(branchNum)
    
    model = Model(branchLs, input_dim)
    model.load_state_dict(weight)

    # set robot urdf
    robot = ET.Element("robot", name='softrobot')

    # set world link & ground joint
    robot = set_ground(robot)
    print(ET.tostring(robot))

    # get joint
    n_joint = len(branchLs)
    for joint in range(n_joint):
        PRset = getattr(model.trans_layer,'joint_'+str(joint+1))
        Trackbool = PRset.Trackbool
        if Trackbool:
            robot = get_joint(PRset,joint,robot,True)
            
        else:
            robot = get_joint(PRset,joint,robot,False)
        
    xmlstr = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="   ")

    with open(args.save_dir, "w") as f:
        f.write(xmlstr)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'save SMInet to urdf')
    args.add_argument('--data_path', default= './output/temp/checkpoint_70.pth',type=str,
                    help='path to data')
    args.add_argument('--save_dir', default= './2Visualize/test_urdf.xml',type=str,
                    help='path to save')
    args = args.parse_args()
    main(args)