## define U-Net model
import torch
import torch.nn as nn


class DicePlusBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets, smooth=1):
        BCE_loss = self.bce_loss(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs*targets).sum()
        dice_coeff = (2*intersection+smooth)/(inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_coeff
        
        loss = dice_loss + BCE_loss
        return loss


class IOU(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth)/(union + smooth)

        return iou
