#%%
import torch
import numpy as np
def VecLoss(vec_tar,vec):
    vec_norm = torch.linalg.norm(vec,dim=1)
    vec = vec/ (vec_norm.unsqueeze(1))

    vec_tar_norm = torch.linalg.norm(vec_tar,dim=1)
    vec_tar = vec_tar/ (vec_tar_norm.unsqueeze(1))

    # check dot product
    loss = vec * vec_tar
    loss = torch.sum(loss)
    loss = -loss
    
    return loss

def Pos_norm2(output, label):
    output = output[:,:,0:3,3]
    output = output.reshape(-1,output.size()[1]*output.size()[2])

    loss = torch.sqrt(torch.nn.MSELoss()(output,label))
    # print(output, label)
    return loss

def q_entropy(q_value):
    loss = torch.distributions.Categorical(q_value).entropy()
    loss = -loss
    return loss
