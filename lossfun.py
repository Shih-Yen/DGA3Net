

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPECTS_multiloss(nn.Module):
    def __init__(self,
        alpha=1, gamma=2, logits=True, reduce=True,
        w_aspect = 1.0, w_cs = 0.0,w_detect=0.0,
        ASPECT_threshold=6):
        super(ASPECTS_multiloss, self).__init__()
        self.w_aspect = w_aspect
        self.w_cs = w_cs
        self.w_detect = w_detect
        self.focalLoss = BinaryFocalLoss_2( alpha, gamma, logits, reduce)
        self.aspect_loss = ASPECTS_loss( alpha, gamma, logits, reduce,dic_threshold=ASPECT_threshold)
        self.detect_loss = HemisphereDetectLoss( 1, gamma, logits, reduce)
        self.cs_loss = HS_consistency_loss()
    def forward(self,x,y):
        focalloss = self.focalLoss(x,y)
        aspect_loss, x_hs = self.aspect_loss(x,y)
        detect_loss, x_hs = self.detect_loss(x,y)
        cs_loss = self.cs_loss(x,x_hs)
        return focalloss + self.w_aspect*aspect_loss + self.w_cs*cs_loss + self.w_detect * detect_loss



class HemisphereDetectLoss(nn.Module):
    def __init__(self,
        alpha=1, gamma=2, logits=True, reduce=True
    ):
        super(HemisphereDetectLoss, self).__init__()
        self.focalLoss = BinaryFocalLoss_2( alpha, gamma, logits, reduce)
    def get_dichotomized_ASPECTS(self,y):
        y = y.view(-1,10,2)
        y_ASPECTS = torch.sum(y,1)
        y_th = torch.where(y_ASPECTS >= 10,1,0).type_as(y).to(y.device)
        return y_th
    def forward(self,x,y):
        y = self.get_dichotomized_ASPECTS(y)
        x = x.view(-1,10,2)
        x = torch.min(x,1)[0]
        return self.focalLoss(x,y), x

class ASPECTS_loss(nn.Module):
    def __init__(self,
        alpha=1, gamma=2, logits=True, reduce=True,device='cuda',
        dic_threshold=10
    ):
        super(ASPECTS_loss, self).__init__()
        self.focalLoss = BinaryFocalLoss_2( alpha, gamma, logits, reduce)
        self.dic_threshold = dic_threshold
        self.HSClassifier = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1).to(device)
    def get_dichotomized_ASPECTS(self,y):
        y = y.view(-1,10,2)
        y_ASPECTS = torch.sum(y,1)
        y_th = torch.where(y_ASPECTS >= self.dic_threshold,1,0).type_as(y).to(y.device)
        return y_th
    def forward(self,x,y):
        y = self.get_dichotomized_ASPECTS(y)
        x = x.view(-1,10,2)
        x = torch.mean(x,1,keepdim=True)
        x = self.HSClassifier(x).view(x.shape[0],-1)
        return self.focalLoss(x,y), x

        
class HS_consistency_loss(nn.Module):
    def forward(self,x_roi,x_hs):
        # if HS pred is normal, panelize if roi pred is abnormal
        # x_roi: [N x 10 x 2]
        # x_hs: [N x 2]
        x_roi = x_roi.view(-1,10,2)
        x_hs = x_hs.view(x_hs.shape[0],1,-1)
        l_roi = torch.relu(-x_roi) 
        l_hs = torch.relu(x_hs) 
        loss = l_roi * l_hs
        return torch.mean(loss)

        
        



class BinaryFocalLoss_2(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(BinaryFocalLoss_2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        inputs = inputs.flatten()
        targets = targets.flatten()
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets,  reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * self.alpha + (1- targets) * (1-self.alpha)
        F_loss = alpha_t * (1-pt)**self.gamma * BCE_loss
        #F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

class LabelSmoothingCrossEntropyOld(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1,**kwargs):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, alpha = [0.5,0.5], smoothing=0.1,**kwargs):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.alpha = torch.tensor(alpha).unsqueeze(0)
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        logprobs = F.log_softmax(x, dim=-1)
        if logprobs.ndim > 2:
            logprobs = logprobs.view(-1,logprobs.shape[-1])
        logprobs =  logprobs *  self.alpha.to(logprobs.device)
        target = target.flatten().type(torch.int64)


        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()