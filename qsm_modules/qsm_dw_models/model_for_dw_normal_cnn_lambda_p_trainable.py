#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 12:25:58 2022

@author: venkatesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:12:19 2021

@author: venkatesh
"""
import torch.nn.functional as F
import os
import copy
import torch
import codecs
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets


#%%


class Dw(nn.Module):
    def __init__(self):
        super(Dw,self).__init__()

        self.lambda_val = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad = True)

        self.p=torch.nn.Parameter(torch.Tensor([1.4]), requires_grad = True)

        self.conv1=nn.Sequential(nn.Conv3d(2,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv3=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv4=nn.Sequential(nn.Conv3d(64,64,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(64),
                            nn.ReLU())
        self.conv5=nn.Sequential(nn.Conv3d(64,2,kernel_size=3,stride=1,padding=1),
                            nn.BatchNorm3d(2))

        

    def forward(self,x):
        x1=self.conv1(x)
        # print(x1.shape)
        x2=self.conv2(x1)
        # print(x2.shape)

        x3=self.conv3(x2)
        # print(x3.shape)

        x4=self.conv4(x3)
        # print(x4.shape)

        x5=self.conv5(x4)    
        # print(x5.shape)

        out = x + x5
        return out
    
#%%


# # creating a model
# model=Dw().cuda()
# # #%%
# device_id=0


# net=Dw();
# net=net.cuda(device_id)
# print(net)

# x=torch.randn(1, 2,176,176,160)
# x=x.cuda(device_id);
# tic()
# with torch.no_grad():
#     y=net(x)
# toc()
# print(y.shape)


# for i in net.parameters():
#     print(i.shape,i.name)
    
# pp=0
# for p in (net.parameters()):
#     nn=1
#     for s in p.size():
#         nn=nn*s
#     pp+=nn      
# print('number of parameters: ',pp)
# #%%
# from fvcore.nn import FlopCountAnalysis, parameter_count
# input_tensor = torch.randn(1, 2,176,176,160)

# # Calculate FLOPs
# flop_counter = FlopCountAnalysis(net, x)
# flops = flop_counter.total()

# # Calculate parameter count
# params = parameter_count(net)

# # Print results
# print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Total Parameters: {params['']:,.0f}")
