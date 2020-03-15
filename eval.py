# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:18:02 2019

@author: Nolan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    # initializers
    def __init__(self, d=16):
        super().__init__()
        self.conv1= nn.Conv2d(3,16,3,1,1)#from 250 to 250
        self.conv2 = nn.Conv2d(16,32,3,1,1)#from 250 to 250
        self.conv3 = nn.Conv2d(32,32,3,1,1)#from 250 to 250
        self.conv3 = nn.Conv2d(32,8,3,1,1)#from 250 to 250
        self.conv4 = nn.Conv2d(8,3,3,1,1)#from 250 to 250
        self.conv5 = nn.Conv2d(3,3,3,1,1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        #print("G " + str(x.size()))
        x = F.relu(self.conv1(x))#from 250 to 250
        x= self.bn3(x)
        x = F.relu(self.conv2(x))#from 250 to 250
        x=self.bn1(x)
        x = F.relu(self.conv3(x))#from 250 to 250
        x=self.bn2(x)
        x = (self.conv4(x))#from 250 to 250      
        x = torch.tanh((x))
        return x
    
#path of the trained model
PATH = 'C://Users/Nolan/Downloads/model(4)'
#instantiate generator
model = Generator()
#load in the model
model.load_state_dict(torch.load(PATH,map_location = torch.device('cpu')))
#put it in eval mode
model.eval()

#turn on the webcam
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


#while the webcam is running
while rval:
    #read from teh webcam
    rval, frame = vc.read()
    #crop the webcam
    frame2=frame[150:400,150:400,:]
    #normalize the pixels
    frame3=(frame2 - np.min(frame2))/np.ptp(frame2)
    #convert to a tensor
    frame4 = torch.tensor(frame3)
    #transpose dimensions
    frame5 = frame4.transpose(0,2)
    #make sure in float32 in tensor
    image=torch.tensor(np.asarray(frame5).astype(np.float32))

    #run webcam through the model
    out=model(image.unsqueeze(0))
    #remove extra dimension
    out2=out.squeeze(0)
    #transpose dimensions
    out3=out2.transpose(0,2).transpose(0,1)
    #remove grad requirement so it can be converted to numpy
    out4=out3.detach().numpy()
    #I believe this is normalizing again? can't remember
    out5=(out4 - np.min(out4))/np.ptp(out4)
    #rotate the output image
    cv2.imshow("preview", np.rot90(out5,k=3,axes=(0,1)))
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")

vc.release()
