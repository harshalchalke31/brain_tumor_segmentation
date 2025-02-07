import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss,self).__init__()
    
    def forward(self,pred,target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred*target).sum()
        return 1- ((2.* intersection +smooth)/(pred.sum()+target.sum(+smooth)))