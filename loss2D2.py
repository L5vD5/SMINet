#%%
from audioop import mul
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

def Pos_norm2(output, label, i):
    output = output[:,:,0:3,3]
    output = output.reshape(-1,output.size()[1]*output.size()[2])

    # multiplyError = torch.Tensor([])
    # for i in range(12):
    #     multiplyError = torch.cat((multiplyError, torch.Tensor([i+1])),0)
    #     multiplyError = torch.cat((multiplyError, torch.Tensor([i+1])),0)
    #     multiplyError = torch.cat((multiplyError, torch.Tensor([i+1])),0)
    # multiplyError = torch.flip(multiplyError, dims=[0])
    # loss = torch.sqrt(torch.nn.MSELoss()(torch.multiply(multiplyError,output),torch.multiply(multiplyError,label)))
    loss = torch.sqrt(torch.nn.MSELoss()(output[:,3*i:3*(i+1)], label[:,3*i:3*(i+1)]))
    # loss = torch.sqrt(torch.nn.MSELoss()(output[0], label[0]))
    # print(output, label)
    return loss

def q_entropy(q_value):
    loss = torch.distributions.Categorical(q_value).entropy()
    loss = -loss
    return loss
