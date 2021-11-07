# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss
from scipy.ndimage import distance_transform_edt as distance


@weighted_loss
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


@LOSSES.register_module()
class CustomDiceLoss(nn.Module):
    """
    Custom Dice Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, class_weight=None, loss_weight=1.0, reduction='mean', loss_name='loss_custom_dice', ignore_index=-100):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(CustomDiceLoss, self).__init__()
        
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._loss_name = loss_name
        
        

    def forward(self, pred, target, **kwargs):
        
        reduction = self.reduction
        
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None
            
        pred_soft = F.softmax(pred, dim=1)
        
        total_dice_loss = 0
        for c in range(0, pred.shape[1]):
            loss_seg_dice = dice_loss(pred_soft[:, c, :, :], target == c)
            total_dice_loss+=loss_seg_dice
        total_dice_loss /= pred.shape[1]
        
        
        
        return total_dice_loss * self.loss_weight
 

        
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
