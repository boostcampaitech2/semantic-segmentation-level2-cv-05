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

# 참고 : https://github.com/JunMa11/SegWithDistMap/blob/5a67153bc730eb82de396ef63f57594f558e23cd/code/train_LA_HD.py#L106

# @weighted_loss
# def focal_tversky_loss(pred,
#               target,
#               valid_mask,
#               alpha,
#               beta,
#               gamma,
#               class_weight=None,
#               ignore_index=255):
    
#     assert pred.shape[0] == target.shape[0]
#     total_loss = 0
#     num_classes = pred.shape[1]
    
#     for i in range(num_classes):
#         if i != ignore_index:
#             ft_loss = binary_focal_tversky_loss(
#                 pred[:, i],
#                 target[..., i],
#                 valid_mask=valid_mask,
#                 alpha=alpha,
#                 beta=beta,
#                 gamma=gamma)
#             if class_weight is not None:
#                 ft_loss *= class_weight[i]
#             total_loss += ft_loss
#     return total_loss / num_classes

# @weighted_loss
# def binary_focal_tversky_loss(pred, target, valid_mask, alpha, beta, gamma, **kwargs):
    
#     assert pred.shape[0] == target.shape[0]
#     pred = pred.reshape(pred.shape[0], -1)
#     target = target.reshape(target.shape[0], -1)
#     valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)
#     epsilon = 1e-6
    
    
#     P_G = torch.sum(pred * target, 1)  # TP
#     P_NG = torch.sum(pred * (1 - target), 1)  # FP
#     NP_G = torch.sum((1 - pred) * target, 1)  # FN
    
#     loss = P_G / (P_G + alpha * P_NG + beta * NP_G + epsilon)
#     loss = torch.pow((1 - loss), 1 / gamma)
    
#     return loss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b][c].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


@weighted_loss
def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    
    delta_s = (seg_soft - gt.float())**2
    s_dtm = seg_dtm ** 2
    g_dtm = gt_dtm ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bcxy, bcxy->bcxy', delta_s, dtm)
    # print(multipled.mean())
    return multipled.mean()
#     total_hd_loss = 0
#     num_classes = seg_soft.shape[1]
#     for c in range(1, num_classes):
#         delta_s = (seg_soft[:,c,...] - gt[:,c,...].float()) ** 2
#         s_dtm = seg_dtm[:,c,...] ** 2
#         g_dtm = gt_dtm[:,c,...] ** 2
#         dtm = s_dtm + g_dtm
#         multipled = torch.einsum('bxy, bxy->bxy', delta_s, dtm)
#         hd_loss = multipled.mean()
#         total_hd_loss+=hd_loss
      
#     return total_hd_loss/(num_classes-1)

    
@LOSSES.register_module()
class HausdorffLoss(nn.Module):
    """
    Hausdorff Distance Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, class_weight=None, loss_weight=1.0, reduction='mean', loss_name='loss_hausdorff', ignore_index=-100):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(HausdorffLoss, self).__init__()
        
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
        
        # ce loss
        ce_loss = F.cross_entropy(pred, target)
        
        pred_soft = F.softmax(pred, dim=1)
        # dicel loss
        total_dice_loss = 0
        for c in range(0, pred.shape[1]):
            loss_seg_dice = dice_loss(pred_soft[:, c, :, :], target == c)
            total_dice_loss+=loss_seg_dice
        total_dice_loss /= pred.shape[1]
        
        
        # hausdorff distance loss
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2)
        
        # compute distance maps and hd loss
        with torch.no_grad():
            # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
            gt_dtm_npy = compute_dtm(one_hot_target.cpu().numpy(), pred_soft.shape)
            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(pred_soft.device.index)
            seg_dtm_npy = compute_dtm(pred_soft.cpu().numpy()>0.5, pred_soft.shape)
            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(pred_soft.device.index)

        total_hd_loss = hd_loss(pred_soft, one_hot_target, seg_dtm=seg_dtm, gt_dtm=gt_dtm)
        # loss = alpha*(ce_loss+total_dice_loss) + (1 - alpha) * total_hd_loss

        
    
        return total_hd_loss
 

        
    
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
    
