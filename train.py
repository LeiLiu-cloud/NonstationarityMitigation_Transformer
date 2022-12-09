# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 21:55:57 2022

@author: lei liu

This is an training exmaple for ViT.py; 
To train SwinT, load SwinT.py and make corresponding modifications for model initialization
"""
import time
import torch
from torch import nn
from ViT import ViT
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import wandb 

# Initialize WandB
wandb.init(name='224x224_no dataaug',
           project = 'ViT',
           entity = 'leiliu',
           config = {'image_size':224,
                           'patch_size':8,
                           'dimension':32,
                           'depth':6,
                           'heads':3,
                           'mlp_dim':128,
                           'learning_rate':1e-6,
                           'epochs':300,
                           'batch_size':32,
                           })

#load data
X = np.load('224cellsX.npy')  #224celss * 5
y = np.load('224cellsY.npy') #max400

# data augmentation
#transfer to torch.tensor
x = torch.from_numpy(X.reshape(3000,1,224,224)).float()
y = torch.from_numpy(y.reshape(-1,1)).float()
# scaler
X_ss = (x-x.mean())/(x.std())
Y_ss = y/400;



class Mydata(Dataset):
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        return image, label
    
    def __len__(self):
        return len(self.data)

dataset = Mydata(X_ss,Y_ss)


train_size = int(0.7*len(dataset))
test_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, test_size]) 

train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True) 
val_dataloader = DataLoader(val_set, batch_size=32, shuffle=True)     

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load model
model = ViT(
            image_size = 224,
            patch_size = 8,
            num_classes = 1,
            dim = 32, 
            depth = 6,
            heads = 3,
            mlp_dim = 128, 
            dropout = 0.0,
            emb_dropout = 0.0
            )

model.to(device)

wandb.watch(model)

     
#define loss function
loss_fn = nn.MSELoss()
loss_fn = loss_fn.cuda()

#define optimizer
lr = 1e-6
optimizer = torch.optim.Adam(model.parameters(),lr=lr, eps=1e-8, weight_decay=1e-4)

#traiing parameters
total_train_step = 0
total_val_step = 0
epoch = 500

#add on W&B
start_time = time.time()

for i in range(epoch):
    print('----------This is the {} epoch training---------'.format(i+1))
    train_loss = 0
    
    model.train()
    for imgs, labels in train_dataloader: #for 1 epoch, 66 steps are needed (train data/ batch size)
    
        imgs, labels = imgs.to(device), labels.to(device)
            
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)

        #optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() #update weights       
        total_train_step = total_train_step + 1 #step means iteration, 1 iter means train the number of (batch size) samples         
        train_loss += loss.item()

    wandb.log({'epoch':i,
               'train_loss':train_loss/len(train_dataloader),
               })
    print('total train loss:{}'.format(train_loss))
    print('time elapsed: {:.4f}s'.format(time.time()-start_time))
    
    
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            total_val_loss = total_val_loss + loss.item() 
            
            total_val_step += 1 
    wandb.log({'epoch':i,
               'val_loss':total_val_loss/len(val_dataloader),
               })      
    print('total val loss:{}'.format(total_val_loss))
    

