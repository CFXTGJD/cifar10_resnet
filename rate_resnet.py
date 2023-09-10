import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import timm

import numpy as np
#from tqdm.notebook import tqdm
from tqdm import tqdm
import glob
import os
import matplotlib.pyplot as plt
import cv2

import torchvision.transforms as transforms
from timm.data.transforms import ToNumpy
import torchvision

import torchvision.utils


import numpy as np
table = None
def load_table(sti_time = 250, spike_len = 100, neuro_id=11):
    '''load the table of corresponding spiking of particular pixel value'''
    global table, spike_mean, spike_std
    table = np.load(f"../Fourier/data/npyfiles/pixel-steps-{neuro_id}-{sti_time}-frates-len{spike_len}.npy").astype(np.float32)
    data = np.load(f'../Fourier/data/npzfiles/rate_mean_and_std_len{spike_len}.npz')
    spike_mean, spike_std = data['mean'], data['std']
    # scale to [0, 1]
    table = (table - table.min())/ (table.max() - table.min())
    print('---table loaded---')

def img2spike(img, spike_len, neuro_id=11):
    '''convert img batch(B, C, H, W) to a spiking batch(T, B, H, W, C)'''
    global table
    if type(table) == type(None):
        load_table(spike_len=spike_len, neuro_id=neuro_id)
    # index = img.detach().cpu().numpy().astype(np.uint8)
    # print(f'shape: {img.shape}') torch.Size([16, 3, 32, 32])
    index = img
    spike = table[index]

    # B, C, H, W, T -> B, H, W, T, C
    spike = spike.transpose(0, 2, 3, 4, 1)
    # print(f'spike shape: {spike.shape}') # (4, 32, 32, 100, 3)

    # normalize
    # 防止std太小
    spike = (spike - spike_mean) / (spike_std + 1e-6)

    # B, H, W, T, C -> T, B, H, W, C
    spike = torch.as_tensor(spike, dtype=torch.float32, device='cpu').permute(3, 0, 1, 2, 4).contiguous()
    
    return spike


batch_size=1024
val_batch_size=1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

spike_len=4
C=3
neuro_id=11

class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img
    
transform = transforms.Compose([
        ToNumpy(),
    ])

train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                        train=True, transform=transform,
                                        download=False)
valid_dataset = torchvision.datasets.CIFAR10(root='../data',
                                        train=False, transform=transform,
                                        download=False)
img, label = train_dataset.__getitem__(1000)
img=img.transpose(1,2,0)
plt.imshow(img)
print(label)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True)#这里之前写的是False，是否有关？


class Classifier(nn.Module):
    def __init__(self,num_classes):
        #define necessary layers
        super().__init__()
        self.num_classes=num_classes
        self.model = timm.create_model(model_name="resnet34",pretrained=True)
        self.model.fc=nn.Linear(self.model.fc.in_features, out_features=num_classes)

    def forward(self,X):
        #define forward pass here
        return F.softmax(self.model(X),dim=-1)
    
model=Classifier(10).to(DEVICE)
print(model(torch.zeros((1,3,256,256)).to(DEVICE)).shape)


optimizer=Adam(lr=0.01, params=model.parameters())

def loss_fn(y_pred, y_true):
    return F.cross_entropy(y_pred,y_true)


def train_one_epoch(dataloader, optimizer, loss_fn, len_dataloader):
    dataloader=tqdm(dataloader)
    L=0
    acc=0
    for i,(x,y) in enumerate(dataloader):
        #print(f"x.shape:{x.shape}")#[64,3,32,32] 第一个64为bs
        spike = img2spike(x, spike_len=spike_len, neuro_id=neuro_id)
        #print(f"spike.shape:{spike.shape}")#spike.shape:torch.Size([4, 64, 32, 32, 3])
        spike=spike.mean(0)
        #print(f"spike.shape:{spike.shape}")#spike.shape:torch.Size([64, 32, 32, 3])
        spike=spike.permute(0,3,1,2)
        #print(f"spike.shape:{spike.shape}")#spike.shape:torch.Size([64, 3, 32, 32])
        x=spike
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        x=x.to(torch.float32)
        #print(x.dtype)
        y_pred=model(x)
        l=loss_fn(y_pred,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        L+=l.item()
        acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1)==y.cpu().detach().numpy().argmax(-1))
    return L/len_dataloader, acc/len_dataloader

def valid_one_epoch(dataloader, loss_fn, len_dataloader):
    dataloader=tqdm(dataloader)
    L=0
    acc=0
    for i,(x,y) in enumerate(dataloader):
        spike = img2spike(x, spike_len=spike_len, neuro_id=neuro_id)
        spike=spike.mean(0)
        spike=spike.permute(0,3,1,2)
        x=spike
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        x=x.to(torch.float32)
        #print(x.dtype)
        y_pred=model(x)
        l=loss_fn(y_pred,y)
        L+=l.item()
        acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1)==y.cpu().detach().numpy().argmax(-1))
    return L/len_dataloader, acc/len_dataloader

prev_valid_acc=0
for epoch in range(100):
    train_loss, train_acc = train_one_epoch(train_dataloader,optimizer,loss_fn,len(train_dataset))
    valid_loss, valid_acc = valid_one_epoch(valid_dataloader,loss_fn,len(valid_dataset))
    print(f"epoch:{epoch} | train loss:{train_loss} | valid loss:{valid_loss} | valid_acc:{valid_acc}")
    if prev_valid_acc<valid_acc:
        print("model saved..!!")
        torch.save(model.state_dict(),"best.pt")
        prev_valid_acc=valid_acc