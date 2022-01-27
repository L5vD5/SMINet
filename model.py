#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *

class PRjoint(nn.Module):
    def __init__(self,Trackbool):
        super(PRjoint, self).__init__()
        self.Trackbool = Trackbool
        from_ = -0.1
        to_ = 0.1
        self.rev_p_offset = nn.Parameter(torch.Tensor(1,3).uniform_(from_,to_))
        self.rev_rpy_offset = nn.Parameter(torch.Tensor(1,3).uniform_(from_,to_))
        self.pri_p_offset = nn.Parameter(torch.Tensor(1,3).uniform_(from_,to_))
        self.pri_rpy_offset = nn.Parameter(torch.Tensor(1,3).uniform_(from_,to_))
        self.rev_axis = nn.Parameter(torch.Tensor(3).uniform_(from_,to_))
        self.pri_axis = nn.Parameter(torch.Tensor(3).uniform_(from_,to_))
        
        if Trackbool:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')
            self.p_track = torch.zeros(1,3).to(device)
            self.rpy_track = torch.zeros(1,3).to(device)
            
            # self.p_track = nn.Parameter(torch.Tensor(1,3).uniform_(-1,1))
            # self.rpy_track = nn.Parameter(torch.Tensor(1,3).uniform_(-1,1))
            

    def forward(self,rev_q, pri_q):
        batch_size = rev_q.size()[0]
        device = rev_q.device

        rev_T_offset = pr2t(self.rev_p_offset, rpy2r(self.rev_rpy_offset))
        pri_T_offset = pr2t(self.pri_p_offset, rpy2r(self.pri_rpy_offset))

        R= rodrigues(self.rev_axis,rev_q)
        p = torch.zeros(batch_size,3).to(device)
        rev_T = pr2t(p,R)
        
        R = torch.tile(torch.eye(3),(batch_size,1,1)).to(device)
        p = torch.outer(pri_q,self.pri_axis)
        
        pri_T = pr2t(p,R)

        if not(self.Trackbool):
            return rev_T_offset, pri_T_offset, rev_T, pri_T

        T_track = pr2t(self.p_track,rpy2r(self.rpy_track))

        return rev_T_offset, pri_T_offset, rev_T, pri_T, T_track
        
class TransformLayer(nn.Module):
    def __init__(self,branchLs):
        super(TransformLayer, self).__init__()

        self.branchLs = branchLs
        n_joint = len(branchLs)

        for joint in range(n_joint):
            Trackbool = branchLs[joint]
            setattr(self,'joint_'+str(joint+1), PRjoint(Trackbool))

    def forward(self,rev_q_value,pri_q_value):
        assert rev_q_value.size() == pri_q_value.size()
        
        branchLs = self.branchLs
        n_joint = len(branchLs)
        batch_size = rev_q_value.size()[0]
        device = rev_q_value.device
        out = torch.tile(torch.eye(4),(batch_size,1,1)).to(device)
        
        TrackingSE3 = torch.tensor([]).reshape(batch_size,-1,4,4).to(device)
        RevSE3 = torch.tensor([]).reshape(batch_size,-1,4,4).to(device)
        PriSE3 = torch.tensor([]).reshape(batch_size,-1,4,4).to(device)

        for joint in range(n_joint):
            rev_q = rev_q_value[:,joint]
            pri_q = pri_q_value[:,joint]
            
            PRset = getattr(self,'joint_'+str(joint+1))

            if branchLs[joint]:
                rev_T_offset, pri_T_offset, rev_T, pri_T, T_track = PRset(rev_q,pri_q)
                out = out @ rev_T_offset @ rev_T
                RevSE3 = torch.cat((RevSE3,out.unsqueeze(1)), dim=1)
                out = out@ pri_T_offset @ pri_T
                PriSE3 = torch.cat((PriSE3,out.unsqueeze(1)), dim=1)

                out_temp = out@T_track
                TrackingSE3 = torch.cat((TrackingSE3,out_temp.unsqueeze(1)), dim=1)
                
            
            else:
                rev_T_offset, pri_T_offset, rev_T, pri_T = PRset(rev_q,pri_q)
                out = out @ rev_T_offset @ rev_T
                RevSE3 = torch.cat((RevSE3,out.unsqueeze(1)), dim=1)
                out = out@ pri_T_offset @ pri_T
                PriSE3 = torch.cat((PriSE3,out.unsqueeze(1)), dim=1)
        
        return TrackingSE3, RevSE3, PriSE3


class q_layer(nn.Module):
    def __init__(self,branchLs,inputdim,n_layers=5):
        super(q_layer, self).__init__()
        self.branchLs = branchLs
        n_joint = len(branchLs)
        LayerList = []
        for _ in range(n_layers):
            # set FC layer
            layer = nn.Linear(inputdim,4*inputdim)
            # torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.normal_(layer.weight, std=0.1)

            # append FC layer
            LayerList.append(layer)

            # add bn layer
            LayerList.append(torch.nn.BatchNorm1d(inputdim*4))

            # add ac layer
            LayerList.append(torch.nn.LeakyReLU())

            # increse dim
            inputdim = inputdim * 4

        for _ in range(n_layers-2):
            # set FC layer
            layer = nn.Linear(inputdim,inputdim//2)
            # torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.normal_(layer.weight, std=0.1)

            # append FC layer
            LayerList.append(layer)

            # add bn layer
            LayerList.append(torch.nn.BatchNorm1d(inputdim//2))
            
            # add ac layer
            LayerList.append(torch.nn.LeakyReLU())

            # reduce dim
            inputdim = inputdim // 2
        layer = nn.Linear(inputdim,2*n_joint)

        # layer = nn.Linear(inputdim, 64)
        # torch.nn.init.normal_(layer.weight, std=0.1)
        # # torch.nn.init.xavier_uniform_(layer.weight)
        # LayerList.append(layer)
        # LayerList.append(torch.nn.LeakyReLU())
        # layer = nn.Linear(64, 64)
        # # torch.nn.init.xavier_uniform_(layer.weight)
        # torch.nn.init.normal_(layer.weight, std=0.1)
        # LayerList.append(layer)
        # LayerList.append(torch.nn.LeakyReLU())
        # layer = nn.Linear(64, 64)
        # # torch.nn.init.xavier_uniform_(layer.weight)
        # torch.nn.init.normal_(layer.weight, std=0.1)
        # LayerList.append(layer)
        # LayerList.append(torch.nn.LeakyReLU())

        # # layer = nn.Linear(64,2*n_joint)
        layer = nn.Linear(inputdim,2*n_joint)
        torch.nn.init.normal_(layer.weight, std=0.1)
        LayerList.append(layer)
        LayerList.append(torch.nn.LeakyReLU())

        self.layers = torch.nn.Sequential(*LayerList)
        

    def forward(self, motor_control):
        branchLs = self.branchLs
        n_joint = len(branchLs)
        out = motor_control
        out = self.layers(out)

        rev_q_value = out[:,:n_joint]
        pri_q_value = out[:,n_joint:]

        return rev_q_value, pri_q_value

class Model(nn.Module):
    def __init__(self, branchNum, inputdim):
        super(Model,self).__init__()
        self.branchNum = branchNum
        branchLs = bnum2ls(branchNum)
        self.q_layer = q_layer(branchLs, inputdim)
        self.trans_layer = TransformLayer(branchLs)

    def forward(self, motor_control):
        rev_q_value, pri_q_value = self.q_layer(motor_control)
        TrackingSE3, RevSE3, PriSE3 = self.trans_layer(rev_q_value, pri_q_value)

        return TrackingSE3, RevSE3, PriSE3




#%%