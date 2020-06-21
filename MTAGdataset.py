# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:30:45 2017

@author: Yongjie Zhu
"""
# load data from MagnaTagATune Dataset
# http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset

import csv
import os
import torch
from torch.utils import data
import audio_processor as ap
import matplotlib.pyplot as plt
from torchvision import  transforms as T

class MusicDataset(data.Dataset):
    """ MagnaTagAtune Dataset """
    def __init__(self,csv_file,root_dir,transforms=None,train=True,test=False,melgrams=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the clips music.
            train /test(callable,optional): divid the dataset into train\valid\test.
            melgrams:(callable,optional): Optional to do STFT in clips.
        """
        # train: audio/1-12
        # valid: audio/13
        # test: audio/14-16
        tags_list=[row for row in csv.reader(open(csv_file),delimiter='\t')]
        self.tags_name=tags_list[0]
        self.test=test
        if self.test:
            parts=['e','f'] # parts14-16
        elif train:
            parts=['0','c'] # parts1:12
        else:
            parts=['d','d'] # parts13
        self.tags_list = [row_tag for row_tag in tags_list[1:] if row_tag[-1][0] >= parts[0] and row_tag[-1][0] <= parts[1]]
        self.root_dir = root_dir
        self.melgrams = melgrams
        
        if transforms is None:
            self.transforms = T.Compose([
                    T.ToTensor()])
    
        
    def __len__(self):
        return len(self.tags_list)-1
    
    def __getitem__(self,idx):
        clips_name = os.path.join(self.root_dir,self.tags_list[idx][-1])
        mel = ap.compute_melgram(clips_name)
        mel = mel[0,:,:,:]
        mel = mel.transpose([1,2,0])
        tags_vec = [int(i) for i in self.tags_list[idx][1:-2]]
        mel = self.transforms(mel)
        tags = torch.Tensor(tags_vec[:120])
        return mel,tags
    
########################################################################
# 1. Load data using torch Loader function.
# trainset validset testset without norm

# trainset
trainset = MusicDataset('E:\Study Place\python\deeplearning&neuroscience/annotations_final.csv',
                        'E:\Study Place\python\deeplearning&neuroscience/audio/')

trainloader = data.DataLoader(trainset,batch_size=4,shuffle=False)

# validset
valideset = MusicDataset('E:\Study Place\python\deeplearning&neuroscience/annotations_final.csv',
                         'E:\Study Place\python\deeplearning&neuroscience/audio/',train=False) 

valideloader = data.DataLoader(valideset,batch_size=4,shuffle=False)

# testset
testset = MusicDataset('E:\Study Place\python\deeplearning&neuroscience/annotations_final.csv',
                       'E:\Study Place\python\deeplearning&neuroscience/audio/',test=True)

testloader = data.DataLoader(testset,batch_size=4,shuffle=False)

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  just like AlexNet

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=(2,1)),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2,4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=(2,1)),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(2,4))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=(2,1)),
            nn.BatchNorm2d(512),
            nn.Sigmoid(),
            nn.MaxPool2d(2,4))
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=(2,1)),
            nn.BatchNorm2d(1024),
            nn.Sigmoid(),
            nn.MaxPool2d(3,5))
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=(3,1)),
            nn.BatchNorm2d(2048),
            nn.Sigmoid(),
            nn.MaxPool2d(4,4))
        self.fc=nn.Linear(1*1*2048,120)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn=CNN()
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum

import torch.optim as optim

criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(cnn.parameters())

# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, clip_data in enumerate(valideloader):
        # get the inputs
        inputs, labels = clip_data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

