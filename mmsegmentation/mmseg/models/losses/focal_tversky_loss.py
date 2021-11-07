# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/LikeLy-Journey/SegmenTron/blob/master/
segmentron/solver/loss.py (Apache-2.0 License)"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ..builder import LOSSES
from .utils import get_class_weight, weighted_loss



@weighted_loss
def focal_tversky_loss(pred,
              target,
              valid_mask,
              alpha,
              beta,
              gamma,
              class_weight=None,
              ignore_index=255):
    
    assert pred.shape[0] == target.shape[0]
    total_loss = 0
    num_classes = pred.shape[1]
    
    for i in range(num_classes):
        if i != ignore_index:
            ft_loss = binary_focal_tversky_loss(
                pred[:, i],
                target[..., i],
                valid_mask=valid_mask,
                alpha=alpha,
                beta=beta,
                gamma=gamma)
            if class_weight is not None:
                ft_loss *= class_weight[i]
            total_loss += ft_loss
    return total_loss / num_classes

@weighted_loss
def binary_focal_tversky_loss(pred, target, valid_mask, alpha, beta, gamma, **kwargs):
    
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
    epsilon = 1e-6
    
    
    P_G = torch.sum(pred * target, 1)  # TP
    P_NG = torch.sum(pred * (1 - target), 1)  # FP
    NP_G = torch.sum((1 - pred) * target, 1)  # FN
    
    loss = P_G / (P_G + alpha * P_NG + beta * NP_G + epsilon)
    loss = torch.pow((1 - loss), 1 / gamma)
    
    return loss
  

    
@LOSSES.register_module()
class FocalTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, class_weight=None, loss_weight=1.0, reduction='mean', loss_name='loss_focaltversky', ignore_index=-100):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weight = get_class_weight(class_weight)
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._loss_name = loss_name
        
        

    def forward(self, pred, target, **kwargs):
        
        reduction = self.reduction
        
        if self.class_weight is not None:
            class_weight = inputs.new_tensor(self.class_weight)
        else:
            class_weight = None
            
            
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()
        
        loss = self.loss_weight * focal_tversky_loss(
            pred,
            one_hot_target,
            valid_mask=valid_mask,
            reduction=reduction,
            class_weight=class_weight,
            alpha = self.alpha,
            beta =  self.beta,
            gamma = self.gamma,
            ignore_index=self.ignore_index)
        return loss
 

        
    
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