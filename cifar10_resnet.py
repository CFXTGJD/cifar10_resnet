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
from timm.data.transforms import ToNumpy, ToTensor
import torchvision

import torchvision.utils


batch_size=1024
val_batch_size=1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE="cpu"
print(DEVICE)


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
# transform = transforms.Compose([
#         ToTensor()
#     ])

train_dataset = torchvision.datasets.CIFAR10(root='../data',
                                        train=True, transform=transform,
                                        download=True)
valid_dataset = torchvision.datasets.CIFAR10(root='../data',
                                        train=False, transform=transform,
                                        download=True)
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

optimizer=Adam(lr=0.001, params=model.parameters())

def loss_fn(y_pred, y_true):
    return F.cross_entropy(y_pred,y_true)

def train_one_epoch(dataloader, optimizer, loss_fn, len_dataloader):
    dataloader=tqdm(dataloader)
    L=0
    acc=0
    for i,(x,y) in enumerate(dataloader):
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        x=x.to(torch.float32)
        # print(x.dtype)
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
        x=x.to(DEVICE)
        y=y.to(DEVICE)
        x=x.to(torch.float32)
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