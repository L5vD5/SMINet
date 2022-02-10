#%%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
class ToyDataset(Dataset):
    def __init__(self,data_path):
        if os.path.isdir(data_path):
            Scenario_ls = os.listdir(data_path)
            self.label = torch.tensor([])
            self.input = torch.tensor([])

            for scenario in Scenario_ls:
                file_path = data_path + '/' + scenario
                rawdata = np.loadtxt(file_path)
                dof = 1
                self.label = torch.cat((self.label,torch.Tensor(rawdata[:,dof:])),0)
                self.input = torch.cat((self.input,torch.Tensor(rawdata[:,:dof])),0)
        if os.path.isfile(data_path):
            file_path = data_path
            # npz load
            rawdata = np.load(data_path)['arr_0']
            # 2d -> 3d
            rawdata = np.insert(rawdata, (3,5,7,9,11,13,15,17,19,21,23,25), 0, axis=1)
            dof = 1
            self.label = torch.Tensor(rawdata[:,dof:])
            self.input = torch.Tensor(rawdata[:,:dof])

            self.label /= 1000

            # shape = self.label[:,0::3].shape

            # self.scaler_label = MinMaxScaler()
            # self.scaler_label.fit(torch.flatten(self.label[:,0::3]).reshape(-1,1))
            # self.label[:,0::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,0::3]).reshape(-1,1)).reshape(shape))
            # self.label[:,0::3] = self.label[:,0::3] - 1

            # self.scaler_label = MinMaxScaler()
            # self.scaler_label.fit(torch.flatten(self.label[:,1::3]).reshape(-1,1))
            # self.label[:,1::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,1::3]).reshape(-1,1)).reshape(shape))

            # self.scaler_label = MinMaxScaler()
            # self.scaler_label.fit(torch.flatten(self.label[:,2::3]).reshape(-1,1))
            # self.label[:,2::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,2::3]).reshape(-1,1)).reshape(shape))

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx], self.label[idx]

class ToyDataloader(DataLoader):
    def __init__(self,data_path, n_workers,batch, shuffle = True):
        self.dataset = ToyDataset(data_path)
        super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

class FoldToyDataset(Dataset):
    def __init__(self,data_path,Foldstart,Foldend):
        self.label = torch.tensor([])
        self.input = torch.tensor([])

        # npz load
        rawdata = np.load(data_path)['arr_0']
        # 2d -> 3d
        rawdata = np.insert(rawdata, (3,5,7,9,11,13,15,17,19,21,23,25), 0, axis=1)
        dof = 1
        self.label = torch.cat((self.label,torch.Tensor(rawdata[:,dof:])),0)
        self.input = torch.cat((self.input,torch.Tensor(rawdata[:,:dof])),0)

        # print(self.label, self.input)
        self.label /= 1000
        # shape = self.label[:,0::3].shape

        # self.scaler_label = MinMaxScaler()
        # self.scaler_label.fit(torch.flatten(self.label[:,0::3]).reshape(-1,1))
        # self.label[:,0::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,0::3]).reshape(-1,1)).reshape(shape))
        # self.label[:,0::3] = self.label[:,0::3] - 1

        # self.scaler_label = MinMaxScaler()
        # self.scaler_label.fit(torch.flatten(self.label[:,1::3]).reshape(-1,1))
        # self.label[:,1::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,1::3]).reshape(-1,1)).reshape(shape))

        # self.scaler_label = MinMaxScaler()
        # self.scaler_label.fit(torch.flatten(self.label[:,2::3]).reshape(-1,1))
        # self.label[:,2::3] = torch.Tensor(self.scaler_label.transform(torch.flatten(self.label[:,2::3]).reshape(-1,1)).reshape(shape))


    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx], self.label[idx]

class FoldToyDataloader(DataLoader):
    def __init__(self,data_path, Foldstart, Foldend, n_workers,batch, shuffle = True):
        self.dataset = FoldToyDataset(data_path,Foldstart,Foldend)
        super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)

#%%