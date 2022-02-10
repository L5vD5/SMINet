#%%
import argparse
import torch
import numpy as np
from dataloaderImage2D2 import *
from modelImage2D2 import Model
from loss2D2 import *
import os
import random
from pathlib import Path
import wandb
import time
from utils.pyart import *
# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def get_q_loss(rev_q_value,pri_q_value):
    q_loss = q_entropy(torch.abs(rev_q_value)) + q_entropy(torch.abs(pri_q_value))
    q_loss = torch.mean(q_loss,dim=0)

    return q_loss

def get_VecLoss(branchNum, TrackingSE3, RevSE3, PriSE3):
    batch_size = TrackingSE3.size()[0]
    device = TrackingSE3.device
    branchLs = bnum2ls(branchNum)
    nJoint = len(branchLs)

    Vec_loss = torch.tensor(0).to(torch.float).to(device)
    currJoint = 1
    prev_tar = torch.tensor([[0,0,0]]).to(torch.float).to(device)
    prev_pri_p = torch.tensor([[0,0,0]]).to(torch.float).to(device)
    prev_rev_p = torch.tensor([[0,0,0]]).to(torch.float).to(device)

    targetNums = TrackingSE3.size()[1]
    for targetNum in range(targetNums):
        curr_tar = t2p(TrackingSE3[:,targetNum])
        vec_tar = curr_tar - prev_tar
        freeJnum = branchNum[targetNum]
        for joint in range(currJoint,currJoint+freeJnum+1):
            rev_p = t2p(RevSE3[:,joint-1])
            vec = (rev_p - prev_pri_p)
            Vec_loss = Vec_loss + VecLoss(vec_tar,vec)
            prev_rev_p = rev_p

            pri_p = t2p(PriSE3[:,joint-1])
            vec = (pri_p - prev_rev_p)
            Vec_loss = Vec_loss + VecLoss(vec_tar,vec)
            prev_pri_p = pri_p

            currJoint = currJoint + 1
    
    Vec_loss = (Vec_loss/batch_size)/nJoint
    return Vec_loss
            
    
def train_epoch(model, optimizers, input, label,Loss_Fn, args, i):
    # forward model
    rev_q_value, pri_q_value = model.q_layer(input)
    TrackingSE3, RevSE3, PriSE3  = model.trans_layer(rev_q_value, pri_q_value)
    
    # get Pos_loss
    Pos_loss = Loss_Fn(TrackingSE3,label,i)

    # get q_loss
    q_loss = get_q_loss(rev_q_value,pri_q_value)
    q_loss = args.q_entropy * q_loss

    # get Vec_loss
    branchNum = model.branchNum
    Vec_loss =  get_VecLoss(branchNum, TrackingSE3, RevSE3, PriSE3)
    Vec_loss = args.Vec_loss * Vec_loss 

    # sum regularizer_loss
    regularizer_loss = q_loss + Vec_loss

    # sum total loss
    total_loss = Pos_loss + regularizer_loss

    optimizers[i].zero_grad()
    total_loss.backward()
    optimizers[i].step()

    return Pos_loss,q_loss,Vec_loss

def test_epoch(model, input, label, Loss_Fn, args):
    # forward model
    rev_q_value, pri_q_value = model.q_layer(input)
    TrackingSE3, RevSE3, PriSE3  = model.trans_layer(rev_q_value, pri_q_value)
    
    # get Pos_loss
    Pos_loss = Loss_Fn(TrackingSE3,label)

    # get q_loss
    q_loss = get_q_loss(rev_q_value,pri_q_value)
    q_loss =  args.q_entropy * q_loss

    # get Vec_loss
    branchNum = model.branchNum
    Vec_loss =  get_VecLoss(branchNum, TrackingSE3, RevSE3, PriSE3)
    Vec_loss = args.Vec_loss * Vec_loss 


    return Pos_loss,q_loss,Vec_loss

def main(args):
    #set logger
    if args.wandb:
        wandb.init(project = args.pname)

    if torch.cuda.is_available():
        #set device
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu:0')

    #set model
    model = Model(args.branchNum, args.input_dim)
    model = model.to(device)

    #load weight when requested
    if os.path.isfile(args.resume_dir):
        weight = torch.load(args.resume_dir)
        model.load_state_dict(weight['state_dict'])
        print("loading successful!")
    else:
        print("Nothing to load, Starting from scratch")

    #set optimizer
    # optimizer = torch.optim.Adam(model.parameters(),lr= args.lr, weight_decay=args.wd, betas=(0.5, 0.9), eps=0.1)
    optimizers = [torch.optim.Adam(model.parameters(),lr= args.lr, weight_decay=args.wd, betas=(0.5, 0.9), eps=1e-4) for i in range(12)]


    #declare loss function
    if args.loss_function == 'Pos_norm2':
        Loss_Fn = Pos_norm2
    else:
        print("Invalid loss_function")
        exit(0)
    

    #assert path to save model
    pathname = args.save_dir
    Path(pathname).mkdir(parents=True, exist_ok=True)

    #set dataloader
    print("Setting up dataloader")
    train_data_loader = FoldToyDataloader(args.data_path, args.Foldstart, args.Foldend, args.n_workers, args.batch_size)
    test_data_loader = FoldToyDataloader(args.data_path, args.Foldend, -1, args.n_workers, args.batch_size)
    
    print("Initalizing Training loop")
    for epoch in range(args.epochs):
        # Timer start
        time_start = time.time()

        # Train
        model.train()
        baseLr = args.lr
        for joint in range(len(args.branchNum)):
            train_loss = 100
            args.lr=baseLr
            i=0
            while(train_loss > 1e-2):#i<100
                i+=1
                if i > 100:
                    i=0
                data_length = len(train_data_loader)
                train_loss = np.array([])
                for iterate, (input,label) in enumerate(train_data_loader):
                    input = input.to(device)
                    label = label.to(device)
                    Pos_loss,q_loss,Vec_loss = train_epoch(model, optimizers, input, label, Loss_Fn, args, joint)
                    total_loss = Pos_loss + q_loss + Vec_loss
                    train_loss = np.append(train_loss, total_loss.detach().cpu().numpy())
                    # args.lr *= 0.999
                    # print('Epoch:{}, Pos_loss:{:.8f}, Q_loss:{:.8f}, Vec_loss:{:.8f}, Progress:{:.2f}%'.format(epoch+1,Pos_loss,q_loss,Vec_loss,100*iterate/data_length), end='\r')

                train_loss = train_loss.mean()
                print('{:d}th joint TrainLoss:{:.8f}, lr:{:.8f}'.format(joint, train_loss, args.lr), end='\r')
            print('')
            freeze_model = getattr(model.q_layer,'rev_pri_q_'+str(joint+1))
            for p in freeze_model.parameters():
                p.requires_grad = False
            # if joint < len(args.branchNum)-1:
            #     next_model = getattr(model.q_layer,'rev_pri_q_'+str(joint+2))
            #     for param_next, param_curr in zip(next_model.parameters(), freeze_model.parameters()):
            #         param_next.data = param_curr.data
            freeze_model = getattr(model.trans_layer,'joint_'+str(joint+1))
            for p in freeze_model.parameters():
                p.requires_grad = False
            # if joint < len(args.branchNum)-1:
            #     next_model = getattr(model.trans_layer,'joint_'+str(joint+2))
            #     for param_next, param_curr in zip(next_model.parameters(), freeze_model.parameters()):
            #         param_next.data = param_curr.data


        
        #Evaluate
        model.eval()
        data_length = len(test_data_loader)
        test_loss = np.array([])
        avg_Pos_loss = np.array([])
        avg_q_loss = np.array([])
        avg_Vec_loss = np.array([])
        # for iterate, (input,label) in enumerate(test_data_loader):
        #     input = input.to(device)
        #     label = label.to(device)
        #     Pos_loss,q_loss,Vec_loss = test_epoch(model, input, label, Loss_Fn, args)
        #     total_loss = Pos_loss# + q_loss + Vec_loss

        #     # metric to plot
        #     test_loss = np.append(test_loss, total_loss.detach().cpu().numpy())
        #     avg_Pos_loss = np.append(avg_Pos_loss, Pos_loss.detach().cpu().numpy())
        #     avg_q_loss = np.append(avg_q_loss, q_loss.detach().cpu().numpy())
        #     avg_Vec_loss = np.append(avg_Vec_loss, Vec_loss.detach().cpu().numpy())
            
        #     print('Testing...{:.8f} Epoch:{}, Progress:{:.2f}%'.format(Pos_loss,epoch+1,100*iterate/data_length) , end='\r')
        
        # test_loss = test_loss.mean()
        # avg_Pos_loss = avg_Pos_loss.mean()
        # avg_q_loss = avg_q_loss.mean()
        # avg_Vec_loss = avg_Vec_loss.mean()
        # print('TestLoss:{:.8f}'.format(test_loss))

        # # Timer end    
        time_end = time.time()
        avg_time = time_end-time_start
        # eta_time = (args.epochs - epoch) * avg_time
        # h = int(eta_time //3600)
        # m = int((eta_time %3600)//60)
        # s = int((eta_time %60))
        # print("Epoch: {}, TestLoss:{:.8f}, eta:{}:{}:{}".format(epoch+1, test_loss, h,m,s))
        
        # Log to wandb
        if args.wandb:
            wandb.log({'TrainLoss':train_loss, 'TestLoss':test_loss, 'TimePerEpoch':avg_time,
            'avg_Pos_loss':avg_Pos_loss, 'q_entropy':avg_q_loss,'Normalized_Vec_loss':avg_Vec_loss/args.Vec_loss,'Vec_weight':args.Vec_loss},step = epoch+1)

        #save model 
        if (epoch+1) % args.save_period==0:
            filename =  pathname + '/checkpoint_{}.pth'.format(epoch+1)
            print("saving... {}".format(filename))
            state = {
                'state_dict':model.state_dict(),
                # 'optimizer':optimizers.state_dict(),
                'branchNum':args.branchNum,
                'input_dim':args.input_dim
            }
            torch.save(state, filename)

        # schedule args.Vec_loss
        # args.Vec_loss = args.Vec_loss * 0.99


if __name__ == '__main__':
    args = argparse.ArgumentParser(description= 'parse for POENet')
    args.add_argument('--batch_size', default= 1024*8, type=int,
                    help='batch_size')
    args.add_argument('--data_path', default= './data/Multi_2dim_log_spiral',type=str,
                    help='path to data')
    args.add_argument('--save_dir', default= './output/temp',type=str,
                    help='path to save model')
    args.add_argument('--resume_dir', default= './output/yb',type=str,
                    help='path to load model')
    args.add_argument('--device', default= '1',type=str,
                    help='device to use')
    args.add_argument('--n_workers', default=0, type=int,
                    help='number of data loading workers')
    args.add_argument('--wd', default= 0.001, type=float,
                    help='weight_decay for model layer')
    args.add_argument('--lr', default= 0.001, type=float,
                    help='learning rate for model layer')
    # args.add_argument('--optim', default= 'adam',type=str,
    #                 help='optimizer option')
    args.add_argument('--loss_function', default= 'Pos_norm2', type=str,
                    help='get list of loss function')
    args.add_argument('--Vec_loss', default= 1, type=float,
                    help='Coefficient for TwistNorm')
    args.add_argument('--q_entropy', default= 0.01, type=float,
                    help='Coefficient for q_entropy')
    args.add_argument('--wandb', action = 'store_true', help = 'Use wandb to log')
    args.add_argument('--input_dim', default= 2, type=int,
                    help='dimension of input')
    args.add_argument('--epochs', default= 100, type=int,
                    help='number of epoch to perform')
    # args.add_argument('--early_stop', default= 50, type=int,
    #                 help='number of n_Scence to early stop')
    args.add_argument('--save_period', default= 1, type=int,
                    help='number of scenes after which model is saved')
    args.add_argument('--pname', default= 'SMINetLoss',type=str,
                    help='Project name')
    args.add_argument('--Foldstart', default= 0, type=int,
                    help='Number of Fold to start')
    args.add_argument('--Foldend', default= 8, type=int,
                    help='Number of Fole to end')
    args.add_argument("--branchNum", nargs="+", default= [0,0,0,0,0,0,0,0,0,0,0,0])
    args = args.parse_args()
    main(args)
#%%
