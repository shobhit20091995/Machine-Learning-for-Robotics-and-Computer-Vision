import torch
import numpy as np
from typing import Optional

def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: Optional[int] = 2, beta: Optional[int] = 4) -> torch.Tensor:
    """
    This function computes the focal loss as described in the CenterNet paper:

    Args:
        - pred (torch.Tensor): predicted network output heatmap
        - gt (torch.Tensor): ground truth heatmap
        - alpha (int): alpha parameter of the focal loss (use the default value)
        - beta (int): beta parameter of the focal loss (use the default value)

    Returns:
        - loss (torch.Tensor): computed focal loss, a torch.Tensor with shape (1) (only one value)
    """

    loss = None

    
    
 
    positive_loss = (1 - pred) ** alpha * torch.log(pred) * (gt == 1)

        
    negative_loss = (1 - gt) ** beta * (pred ** alpha) * torch.log(1 - pred) *(gt != 1)
    
    N = (gt == 1).float().sum()

    total_loss = positive_loss.sum() + negative_loss.sum()
    loss = -(total_loss) / N
    

    

    

    

    return loss

def smooth_l1_loss(pred: torch.Tensor, gt: torch.Tensor, sz_mask: np.ndarray) -> torch.Tensor:
    """
    This function computes the focal loss as described in the CenterNet paper:

    Args:
        - pred (torch.Tensor): predicted network output sizemap
        - gt (torch.Tensor): ground truth sizemap
        - sz_mask (numpy.ndarray): mask with the index to compute the loss (only the objects centers)

    Returns:
        - loss (torch.Tensor): computed smooth l1 loss, a torch.Tensor with shape (1) (only one value)
    """

    loss = None
    pred_masked = pred[sz_mask]
    gt_masked = gt[sz_mask]
    
    #N = sz_mask.float().sum()
    #sz_mask = torch.from_numpy(sz_mask) 


    # Compute the smooth L1 loss
    loss = torch.nn.functional.smooth_l1_loss(pred_masked, gt_masked) 
    
    return loss

def centerloss(pred_ht, pred_sz, gt_ht, gt_sz, sz_mask, loss_weights):
    # Binary mask loss
    ht_loss = focal_loss(pred_ht, gt_ht)

    # Regression L1 loss
    pred_sz = pred_sz.permute(0,2,3,1)
    gt_sz = gt_sz.permute(0,2,3,1)
    sz_loss = smooth_l1_loss(pred_sz, gt_sz, sz_mask)

    # Sum
    loss = loss_weights[0] * ht_loss + loss_weights[1] * sz_loss

    return loss, ht_loss , sz_loss
