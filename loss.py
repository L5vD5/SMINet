#%%
import torch

def VecLoss(vec_tar,vec):
    vec_norm = torch.linalg.norm(vec,dim=1)
    vec = vec/ (vec_norm.unsqueeze(1))

    vec_tar_norm = torch.linalg.norm(vec_tar,dim=1)
    vec_tar = vec_tar/ (vec_tar_norm.unsqueeze(1))

    # check dot product
    res = vec * vec_tar
    res = torch.sum(res)
    
    return res

def Pos_norm2(output, label):
    output = output[:,:,0:3,3]
    output = output.reshape(-1,output.size()[1]*output.size()[2])
    loss = torch.nn.MSELoss()(output,label)

    return loss

def q_entropy(q_value):
    loss = torch.distributions.Categorical(q_value).entropy()
    loss = -loss
    return loss
