# -*- coding: utf-8 -*-
"""
Created on Sat Sep 4, 2021 - 16:50
Custom Loss functions applied for algorithms predicting asset returns with Pytorch
A similar repository with Tensorflow 2.0 custom loss functions will be soon published

This file presents an abstract of the code used for preparing a publication submitted to peer-reviewed journal
and available on SSRN https://ssrn.com/abstract=3973086  

ATTENTION : Code presented is NOT optimized but rather presented pedagogically

Use of the code without prior reading the article might be misleading or inadequate.    
@author: JDE65 (Github)
j.dessain@navagne.com   ///  j.dessain@ieseg.fr
www.navagne.com
All rights reserved - Copyright Navagne (2021)
"""

# ====  PART 0. Installing libraries ============
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###
### Custom loss functions based on MSE "nn.MSELoss()" in torch
###
    
### Custom loss function AdjLoss1 of the article
# In the article : alpha is 2 or 1.5 
class AdjMSELoss1(nn.Module):
    def __init__(self):
        super(AdjMSELoss1, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        alpha = 2
        loss = (outputs - labels)**2
        adj = torch.mul(outputs, labels)
        adj[adj>0] = 1 / alpha
        adj[adj<0] = alpha
        loss = loss * adj
        return torch.mean(loss)
 
    
### Custom loss function AdjLoss2 of the article
# Sigmoid-like adjustment
# In the article : beta is 2.5 or 2.25 
class AdjMSELoss2(nn.Module):
    def __init__(self):
        super(AdjMSELoss2, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        beta = 2.5
        loss = (outputs - labels)**2
        adj_loss = beta - (beta - 0.5) / (1 + torch.exp(10000 * torch.mul(outputs, labels)))
        loss = beta * loss /(1+adj_loss) 
        return torch.mean(loss)


### Custom loss function AdjLoss3 of the article
# ReLu-like adjustment
# In the article : gamma is 0.1 
class AdjMSELoss3(nn.Module):
    def __init__(self):
        super(AdjMSELoss3, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        gamma = 0.1
        loss = (outputs - labels)**2
        adj = torch.mul(outputs, labels)
        adj[adj>0] = gamma
        adj[adj<0] = 1 + gamma
        loss = loss * adj
        return torch.mean(loss)

###
### Custom loss functions based on MAE "nn.L1Loss()" (instead of MSE)
###

### Custom loss function AdjLoss1a  
# In the article : alpha is 2 or 1.5 
class AdjMSELoss1a(nn.Module):
    def __init__(self):
        super(AdjMSELoss1a, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        alpha = 2
        loss = (outputs - labels)**2
        adj = torch.mul(outputs, labels)
        adj[adj>0] = 1 / alpha
        adj[adj<0] = alpha
        loss = loss * adj
        return torch.mean(loss)
 
    
### Custom loss function AdjLoss2a 
# Sigmoid-like adjustment
# In the article : beta is 2.5 or 2.25 
class AdjMSELoss2a(nn.Module):
    def __init__(self):
        super(AdjMSELoss2a, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        beta = 2.5
        loss = (outputs - labels)**2
        adj_loss = beta - (beta - 0.5) / (1 + torch.exp(10000 * torch.mul(outputs, labels)))
        loss = beta * loss /(1+adj_loss) 
        return torch.mean(loss)


### Custom loss function AdjLoss2a
# ReLu-like adjustment
# In the article : gamma is 0.1 
class AdjMSELoss3a(nn.Module):
    def __init__(self):
        super(AdjMSELoss3a, self).__init__()
                
    def forward(self, outputs, labels):
        outputs = torch.squeeze(outputs)
        gamma = 0.1
        loss = (outputs - labels)**2
        adj = torch.mul(outputs, labels)
        adj[adj>0] = gamma
        adj[adj<0] = 1 + gamma
        loss = loss * adj
        return torch.mean(loss)
