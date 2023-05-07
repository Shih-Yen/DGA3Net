import random
import os
from turtle import forward
import torch
import numpy as np
import torch.nn as nn
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True


class ReverseModel(nn.Module):
    # reverse the prediciton
    def __init__(self,model):
        super().__init__()
        self.model = model
    def forward(self,x):
        x =  self.model(x).squeeze(-1)
        return torch.sigmoid(-x)
        
def mask_to_bounding_box(mask):
    # convert mask to the mask with bounding boxes
    # input : numpy array, shape: (N, ch, W, H, D)
    img_ndim = mask.ndim - 2
    mask_shape = np.array(mask.shape)
    bbox = np.ones_like(mask)
    for d in range(2,mask.ndim):
        bbox_idx = mask
        max_axes =list(set(range(2,mask.ndim)) - {d})
        bbox_idx = bbox_idx.max(tuple(max_axes),keepdims=True)
        repeat_size = np.ones_like(mask_shape)
        repeat_size[np.array(max_axes)] = mask_shape[np.array(max_axes)]
        dim_mask = np.tile(np.where(bbox_idx>0.1,1.0,0.0),repeat_size)
        bbox = bbox * dim_mask
        pass
    return bbox


        
            



