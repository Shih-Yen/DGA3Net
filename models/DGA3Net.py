import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .layers import * 
poolfun_dict = {
    0: nn.MaxPool3d,
    1: nn.AvgPool3d,
}

class DGA3Net(nn.Module):
    def __init__(self,requires_grad = True,alpha=0.9,
        out_chs=[8,16, 32],block_layers=[2,2,3],block_ds=[True,True,True],
        iso_conv_fun=SymGroupConv3d,
        iso_agg_fun = torch.max,
        use_flip=True,
        dilation=1,
        pool_fun=nn.AvgPool3d,
        dummy_pooltype=1,
        pool_w = 4,
        conv_use_SE=False,
        SE_kernel_w=1,
        SE_use_softmax=True,
        SE_bias=False,
        SE_n_latent_layers=0,
        agg_all_axes=False,
        agg_before_layer=0,
        agg_BN=True,
        bilateral_feature=True,
        bilateral_before_layer=2,
        MLP_ch=[128,32],
        num_class=1,
        ):
        super().__init__()
        # convert 0/1 to bool
        conv_use_SE = bool(conv_use_SE)
        SE_use_softmax = bool(SE_use_softmax)
        SE_bias = bool(SE_bias)
        agg_all_axes = bool(agg_all_axes)
        agg_BN = bool(agg_BN)
        #
        self.pool_w = pool_w
        self.agg_all_axes = agg_all_axes
        self.bilateral_feature = bilateral_feature
        self.num_class = num_class
        if agg_all_axes:
            agg_before_layer = 0
            SE_axis=None
        else:
            SE_axis = 2

        assert len(out_chs) == len(block_layers)
        in_chs = [1] + out_chs[:-1]
        block_list = [ ResSymGroupConv(in_chs[i],out_chs[i],
            n_layers=block_layers[i],ds=block_ds[i],dilation=dilation,pool_fun=pool_fun,alpha=alpha,use_SE=conv_use_SE,
            conv_fun=iso_conv_fun, agg_fun=iso_agg_fun,use_flip=use_flip) 
            for i in range(len(out_chs))]
        self.features = nn.Sequential(*block_list)
        # dummy CNN for subsampling the mask
        num_ds = sum([int(x) for x in block_ds])
        dummy_poolfun = poolfun_dict[dummy_pooltype]
        dummy_list = [dummy_poolfun( kernel_size=(3,3,1), padding = (1,1,0),stride=(2,2,1)) for i in range(num_ds)]
        self.dummy = nn.Sequential(*dummy_list)
        self.pool = nn.AdaptiveMaxPool3d((pool_w,pool_w,None))
        self.SEz = nDSELayer3D(out_chs[-1],axis=SE_axis,
            kernel_w=SE_kernel_w,use_softmax=SE_use_softmax,
            bias=SE_bias,n_latent_layers=SE_n_latent_layers
            )
        SE_chs = [out_chs[-1]] + MLP_ch
        if agg_BN:
            self.agg_BN_layer = nn.BatchNorm3d(SE_chs[agg_before_layer])
        else:
            self.agg_BN_layer = nn.Sequential()

            
        MLP_in_ch = [out_chs[-1]] + MLP_ch
        MLP_out_ch = MLP_ch + [num_class]
        if bilateral_feature:
            MLP_in_ch[bilateral_before_layer] *= 2
            MLP_out_ch[-1] *= 2
            

        
        layers = [
            nn.Sequential(      
                nn.Conv3d(MLP_in_ch[0], MLP_out_ch[0], kernel_size=(pool_w,pool_w,1)),
                nn.ReLU(True),
                nn.Dropout(),
                ),
            nn.Sequential(      
                nn.Conv3d(MLP_in_ch[1], MLP_out_ch[1], kernel_size=(1,1,1)),
                nn.ReLU(True),
                nn.Dropout(),
                ),
            nn.Conv3d(MLP_in_ch[2], MLP_out_ch[2], kernel_size=(1,1,1))
        ]
        self.encoder = nn.Sequential(*layers[:agg_before_layer])
        self.encoder2 = nn.Sequential(*layers[agg_before_layer:bilateral_before_layer])
        self.classifier = nn.Sequential(*layers[bilateral_before_layer:])


    def crop(self,x,mx,my,mz=None,roi=0):
        mx = mx[roi,:]
        my = my[roi,:]
        xi = x[:,mx,:,:]
        xi = xi[:,:,my,:].unsqueeze(0)
        #xi = xi[:,:,:,mz]
        x = self.pool(xi)
        return x
    
    def get_preds(self,x,mask):
        
        mx = mask.max(dim=-1)[0].max(dim=-1)[0].type(torch.bool)
        my = mask.max(dim=-1)[0].max(dim=-2)[0].type(torch.bool)
        mz = mask.max(dim=-2)[0].max(dim=-2)[0].type(torch.bool)
        n_slices = torch.sum(mz,dim=-1)
        out_list = []
        for i in range(mask.shape[0]):
            xi = x[i,...] # 1 * C * X * Y * Z
            xx = [ self.crop(xi,mx[i,...],my[i,...],roi=i_roi)  for i_roi in range(mask.shape[1])]
            xx = torch.stack(xx,dim=-2)
            mbox = [ self.crop(mask[i,i_roi,...].unsqueeze(0),mx[i,...],my[i,...],roi=i_roi)  for i_roi in range(mask.shape[1])]
            mbox = torch.stack(mbox,dim=-2)
            xx *= mbox
            
            out_list.append(xx)
            
        out = torch.cat(out_list,dim=0)
        out = out.permute(0,4,1,2,3,5).contiguous()
        out = out.view(-1,*out.shape[2:])
        mz = mz.view(-1,mz.shape[-1]).unsqueeze(1).unsqueeze(-2).unsqueeze(-2)
        #### aggregate feature
        # intermediate encoding
        out = self.encoder(out)
        # aggregate latent feature over Z axis
        out =self.SEz(out,mz)
        out = torch.sum(out,-1,keepdim=True) / n_slices.flatten().view(-1,1,1,1,1)
        out = self.agg_BN_layer(out) # if specified, do BN after aggregation
        out = self.encoder2(out)
        if self.bilateral_feature:
            out = out.view(x.shape[0]*10,-1,*out.shape[2:])
        # the rest of the classifier
        if not hasattr(self,'num_class'):
            self.num_class = 1
        output = self.classifier(out).view(x.shape[0],-1,self.num_class)
        return output
        

    def forward(self, x):
        with torch.no_grad():
            mask = x[:,2:,...]
            mask = self.dummy(mask)
            x = x[:,0,...].unsqueeze(1)
        img = x
        x = self.features(x)        
        output = self.get_preds(x,mask)
        return output



class ResSymGroupConv(nn.Module):
    def __init__(self,in_ch,out_ch,
        n_layers=2,ds=True,dilation=1,pool_fun=nn.AvgPool3d,alpha=0.9,use_SE=False,
        conv_fun=SymGroupConv3d,agg_fun=torch.max,use_flip=False):
        super().__init__()
        self.conv1 = conv_fun(in_ch, out_ch, kernel_size=(3,3,3), stride=(2,2,1),
            padding = (dilation,dilation,1),dilation=(dilation,dilation,1),agg_fun=agg_fun,use_flip=use_flip)
        self.alpha = alpha
        self.use_SE = use_SE
        convs = []
        for i in range(n_layers):
            convs.append(
                nn.Sequential(
                    conv_fun(out_ch, out_ch, kernel_size=(3,3,3), padding = (dilation,dilation,1),dilation=(dilation,dilation,1),agg_fun=agg_fun,use_flip=use_flip),
                    nn.LeakyReLU(),
                )
            )
        self.convs = nn.Sequential(*convs)
        if ds:
            self.pool = nn.AvgPool3d(kernel_size=(3,3,1),stride=(2,2,1),padding = (1,1,0))
        else:
            self.pool = nn.Sequential()
        if use_SE:
            #self.SE =  SELayer3D(out_ch, reduction=math.floor(math.sqrt(out_ch)))
            self.SE = nDSELayer3D(out_ch,axis=2, reduction=math.floor(math.sqrt(out_ch)))
        self.BN =  nn.BatchNorm3d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        #x = self.pool(x)
        #x1 = self.pool(x1)
        x = self.convs(x1)
        x = self.alpha * x + (1 - self.alpha) * x1
        if self.use_SE:
            x =  self.SE(x)
        x = self.BN(x)
        return x
            
