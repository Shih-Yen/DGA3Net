
from sympy import re
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import *



class BaggedModels(nn.Module):
    def __init__(self,base_model,checkpoints,
        adaptive_threshold=False,
        requires_grad = False,use_SWA=True):
        super().__init__()
        self.use_SWA = use_SWA
        self.base_model_class = base_model
        self.adaptive_threshold = adaptive_threshold
        out = [self.load_a_model(checkpoint) for checkpoint in  checkpoints]
        mdl_bag = [x[0] for x in out]
        thresholds = [x[1] for x in out]
        if adaptive_threshold:            
            thresholds =  torch.FloatTensor(thresholds)
            thresholds = torch.logit(thresholds)
        else:
            thresholds =  torch.zeros((len(checkpoints)))
        self.thresholds = nn.parameter.Parameter(thresholds,requires_grad=False)

        self.mdl_bag = nn.ModuleList(mdl_bag)

    def load_a_model(self,checkpoint):
        checkpoint_state = torch.load(checkpoint)
        if self.use_SWA:
            swa_model = AveragedModel(self.base_model_class())
            best_model = swa_model
            best_model.load_state_dict(checkpoint_state['swa_dict'])
            best_th = checkpoint_state['test_th_best'][-1]
        else:   
            try:
                #best_model.load_state_dict(checkpoint_state['best_model_wts'])
                #best_model = checkpoint_state['model']
                if isinstance(checkpoint_state['best_model_wts'],self.base_model_class):
                    best_model	=	checkpoint_state['best_model_wts']
                else:
                    best_model = self.base_model_class()
                    best_model.load_state_dict(checkpoint_state['best_model_wts'])
                
            except:
                swa_model = AveragedModel(self.base_model_class())
                best_model = swa_model
                best_model.load_state_dict(checkpoint_state['best_model_wts'])
            
            best_th = checkpoint_state['test_th_best']['best_epoch']
        return best_model, best_th

    def forward(self,x,return_list=False):
        logit_list = torch.stack([mdl(x) for mdl in self.mdl_bag],dim=0)
        logit_list = logit_list -    self.thresholds.view(-1,1,1,1)
        logit = torch.mean(logit_list,dim=0)
        if  return_list:           
            return logit, logit_list

        else:
            return logit



