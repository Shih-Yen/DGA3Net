

from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single
from numpy import isin
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class GroupSELayer3D(nn.Module):
    def __init__(self, channel, reduction=2,groups=1,no_sigmoid=False,dropout_p=0):
        super(GroupSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if not no_sigmoid:
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False,groups=groups),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False,groups=groups),
                nn.Sigmoid()
            )
        else:  
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False,groups=groups),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False,groups=groups),
            )
    
    def get_att(self, x):
        #b, c, _, _,_ = x.size()
        y = self.avg_pool(x)
        y = self.att(y)
        return y

    def forward(self, x):
        return x * self.get_att(x)
        
class GroupChannelAttLayer3D(GroupSELayer3D):
    def forward(self, x):
        return self.get_att(x)
        
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=2,no_sigmoid=False,dropout_p=0):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if not no_sigmoid:
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(channel // reduction, channel, bias=False),
                nn.Sigmoid()
            )
        else:  
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(channel // reduction, channel, bias=False),
            )
    
    def get_att(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return y

    def forward(self, x):
        return x * self.get_att(x)
        
class ChannelAttLayer3D(SELayer3D):
    def forward(self, x):
        return self.get_att(x)
        

class nDSELayer3D_alt(nn.Module):
    def __init__(self, channel, reduction=2,
        kernel_w=3,axis=None,dropout_p=0,use_softmax=False,
        bias=True,
        n_latent_layers=2):
        super(nDSELayer3D_alt, self).__init__()        
        self.use_softmax = use_softmax
        self.axis = axis
        self.ch_att_layer = ChannelAttLayer3D(channel,reduction,no_sigmoid=True,dropout_p=dropout_p)
        self.sp_att_layer = SpatialAttLayer3D_alt(
            channel,reduction,no_sigmoid=True,kernel_w=kernel_w,axis=axis,dropout_p=dropout_p,
            bias=bias,n_latent_layers=n_latent_layers)

    def get_att(self, x,mask=None):
        ch_att =  self.ch_att_layer(x)
        sp_att =  self.sp_att_layer(x)
        y = ch_att + sp_att
        #Nonlinear actication (sigmoid or softmax)
        if self.use_softmax:
            if self.axis is not None: #softmax over the specified axis
                y = F.softmax(y,dim=2+ self.axis)
                
            else:#softmax over all axes
                y = F.softmax(y.view(y.shape[:2],-1),dim=-1).view(y.shape)

        else:
            y = torch.sigmoid(y)

        if mask is not None:
            y = y * mask
        return y
        
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
        

class nDSELayer3D(nn.Module):
    def __init__(self, channel, reduction=2,
        kernel_w=3,axis=None,dropout_p=0,use_softmax=False,
        bias=True,
        n_latent_layers=2):
        super(nDSELayer3D, self).__init__()        
        self.use_softmax = use_softmax
        self.axis = axis
        self.ch_att_layer = ChannelAttLayer3D(channel,reduction,no_sigmoid=True,dropout_p=dropout_p)
        self.sp_att_layer = SpatialAttLayer3D(
            channel,reduction,no_sigmoid=True,kernel_w=kernel_w,axis=axis,dropout_p=dropout_p,
            bias=bias,n_latent_layers=n_latent_layers)

    def get_att(self, x,mask=None):
        ch_att =  self.ch_att_layer(x)
        sp_att =  self.sp_att_layer(x)
        y = ch_att + sp_att
        #Nonlinear actication (sigmoid or softmax)
        if self.use_softmax:
            if self.axis is not None: #softmax over the specified axis
                y = F.softmax(y,dim=2+ self.axis)
                
            else:#softmax over all axes
                y = F.softmax(y.view(y.shape[:2],-1),dim=-1).view(y.shape)

        else:
            y = torch.sigmoid(y)

        if mask is not None:
            y = y * mask
        return y
        
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
        

class nDAttLayer3D(nDSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)



class nDSELayer3D_alt2(nDSELayer3D):
    def __init__(self, channel, reduction=2,
        kernel_w=3,axis=None,dropout_p=0,use_softmax=False,
        bias=True,
        n_latent_layers=2):
        super(nDSELayer3D_alt2, self).__init__(channel, reduction=reduction,
        kernel_w=kernel_w,axis=axis,dropout_p=0,use_softmax=use_softmax,
        bias=bias, n_latent_layers=2)
        self.ch_att_layer = ChannelAttLayer3D(channel,reduction,no_sigmoid=True,dropout_p=dropout_p)
        self.sp_att_layer = SpatialAttLayer3D_alt2(
            channel,reduction,no_sigmoid=True,kernel_w=kernel_w,axis=axis,dropout_p=dropout_p,
            bias=bias,n_latent_layers=n_latent_layers)

class nDAttLayer3D_alt2(nDSELayer3D_alt2):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)


class GroupnDSELayer3D(nDSELayer3D):
    def __init__(self, channel, reduction=2,
        kernel_w=3,axis=None,dropout_p=0,use_softmax=False,
        bias=True,groups=1,
        n_latent_layers=2):
        super(nDSELayer3D, self).__init__()        
        self.use_softmax = use_softmax
        self.axis = axis
        self.ch_att_layer = GroupChannelAttLayer3D(channel,reduction,no_sigmoid=True,dropout_p=dropout_p,groups=groups)
        self.sp_att_layer = SpatialAttLayer3D(
            channel,reduction,no_sigmoid=True,kernel_w=kernel_w,axis=axis,dropout_p=dropout_p,
            bias=bias,n_latent_layers=n_latent_layers)

class GroupnnDAttLayer3D(GroupnDSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)

        
class SpatialSELayer3D_alt(nn.Module):
    def __init__(self,channel,
        reduction=2,
        no_sigmoid=False,kernel_w=3,
        axis=2,dropout_p=0,
        bias=True,
        n_latent_layers=2,
        ):
        super(SpatialSELayer3D_alt, self).__init__()
        if axis == None:
            pool_w = [None,None,None]
        else:
            pool_w = [1,1,1]
            pool_w[axis]= None
        pool_w = tuple(pool_w)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        #pad_size = tuple([math.floor(kernel_w)/2,math.floor(kernel_w)/2,1])

        layers = [
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
        for i in range(n_latent_layers): 
            layers += [
                nn.Conv3d(channel // reduction, channel // reduction,kernel_size=kernel_w, padding='same', bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
               
        layers.append(nn.Conv3d(channel // reduction, 1, kernel_size=1, bias=bias))
        if not no_sigmoid:
            layers.append(nn.Sigmoid())
        self.att = nn.Sequential(*layers)
        
    def get_att(self, x,mask=None):
        y = self.avg_pool(x)
        y = self.att(y)
        
        if mask is None:
            return y
        else:
            return y * mask

    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
class SpatialAttLayer3D_alt(SpatialSELayer3D_alt):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)
        
class SpatialSELayer3D_alt2(nn.Module):
    def __init__(self,channel,
        reduction=2,
        no_sigmoid=False,kernel_w=3,
        axis=2,dropout_p=0,
        bias=True,
        n_latent_layers=2,
        ):
        super(SpatialSELayer3D_alt2, self).__init__()
        if axis == None:
            pool_w = [None,None,None]
        else:
            pool_w = [1,1,1]
            pool_w[axis]= None
        pool_w = tuple(pool_w)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        #pad_size = tuple([math.floor(kernel_w)/2,math.floor(kernel_w)/2,1])

        if n_latent_layers == 0:
            layers = [
                    nn.Conv3d(channel, 1, kernel_size=kernel_w, bias=bias, padding='same'),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p)]
        else:

            layers = [
                    nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=bias),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p)]
            for i in range(n_latent_layers-1): 
                layers += [
                    nn.Conv3d(channel // reduction, channel // reduction,kernel_size=kernel_w, padding='same', bias=bias),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_p)]
                
            layers.append(nn.Conv3d(channel // reduction, 1, kernel_size=kernel_w, padding='same',  bias=bias))
        if not no_sigmoid:
            layers.append(nn.Sigmoid())
        self.att = nn.Sequential(*layers)
        
    def get_att(self, x,mask=None):
        y = self.att(x)
        y = self.avg_pool(y)
        
        if mask is None:
            return y
        else:
            return y * mask

    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
class SpatialAttLayer3D_alt2(SpatialSELayer3D_alt2):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)
        
      
        
class SpatialSELayer3D(nn.Module):
    def __init__(self,channel,
        reduction=2,
        no_sigmoid=False,kernel_w=3,
        axis=2,dropout_p=0,
        bias=True,
        n_latent_layers=2,
        ):
        super(SpatialSELayer3D, self).__init__()
        if axis == None:
            pool_w = [None,None,None]
        else:
            pool_w = [1,1,1]
            pool_w[axis]= None
        pool_w = tuple(pool_w)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        kernel_size = [1,1,1]
        if axis is not None:
            kernel_size[axis]= kernel_w
        kernel_size = tuple(kernel_size)
        #pad_size = tuple([math.floor(kernel_w)/2,math.floor(kernel_w)/2,1])

        layers = [
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
        for i in range(n_latent_layers): 
            layers += [
                nn.Conv3d(channel // reduction, channel // reduction,kernel_size=kernel_size, padding='same', bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
               
        layers.append(nn.Conv3d(channel // reduction, 1, kernel_size=1, bias=bias))
        if not no_sigmoid:
            layers.append(nn.Sigmoid())
        self.att = nn.Sequential(*layers)
        
    def get_att(self, x,mask=None):
        y = self.avg_pool(x)
        y = self.att(y)
        
        if mask is None:
            return y
        else:
            return y * mask

    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
       
       
class SpatialAttLayer3D(SpatialSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)
        
        
class SpatialSELayer3D_allAxes(nn.Module):
    def __init__(self, channel, reduction=2, no_sigmoid=False,kernel_w=3,dropout_p=0,
        bias=True,
        n_latent_layers=2):
        super(SpatialSELayer3D_allAxes, self).__init__()
        kernel_size = tuple([kernel_w,kernel_w,kernel_w])
        #pad_size = tuple([math.floor(kernel_w)/2,math.floor(kernel_w)/2,1])

        layers = [
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
        for i in range(n_latent_layers): 
            layers += [
                nn.Conv3d(channel // reduction, channel // reduction,kernel_size=kernel_size, padding='same', bias=bias),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
               
        layers.append(nn.Conv3d(channel // reduction, 1, kernel_size=1, bias=False))
        if not no_sigmoid:
            layers.append(nn.Sigmoid())
        self.att = nn.Sequential(*layers)
        
    def get_att(self, x,mask=None):
        y = self.att(x)
        
        if mask is None:
            return y
        else:
            return y * mask

    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
        

class SpatialAttLayer3D_allAxes(SpatialSELayer3D_allAxes):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)

class SpatialSELayer3D_Old(nn.Module):
    def __init__(self, channel, reduction=2,input_size=(4,4,25),axis=2,no_sigmoid=False):
        super(SpatialSELayer3D, self).__init__()
        self.input_size = input_size
        self.axis = axis
        pool_w = [1,1,1]
        pool_w[axis]= None
        pool_w = tuple(pool_w)
        kernel_size = [1,1,1]
        kernel_size[axis] = input_size[axis]
        kernel_size = tuple(kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        if not no_sigmoid:
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel,kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:  
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel,kernel_size=1, bias=False)
            )

    def get_att(self, x,mask=None):
        y = self.avg_pool(x)
        y = self.att(y)
        #view_shape = [-1,1,1,1,1]
        #view_shape[self.axis+2] = self.input_size[self.axis]
        #y = y.view(*view_shape)
        if mask is None:
            return y
        else:
            return y * mask
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)

class SpatialSELayer3D_Old2(nn.Module):
    def __init__(self, channel, reduction=2,input_size=(4,4,25),axis=2,no_sigmoid=False):
        super(SpatialSELayer3D, self).__init__()
        self.input_size = input_size
        self.axis = axis
        pool_w = [1,1,1]
        pool_w[axis]= None
        pool_w = tuple(pool_w)
        kernel_size = [1,1,1]
        kernel_size[axis] = input_size[axis]
        kernel_size = tuple(kernel_size)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        if not no_sigmoid:
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel,kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:  
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, input_size[axis],kernel_size=1, bias=False)
            )

    def get_att(self, x,mask=None):
        y = self.avg_pool(x)
        y = self.att(y)
        view_shape = [-1,1,1,1,1]
        view_shape[self.axis+2] = self.input_size[self.axis]
        y = y.view(*view_shape)
        if mask is None:
            return y
        else:
            return y * mask
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)



class IsoConv3d(nn.Conv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        agg_fun = torch.max,
        **kwargs
    ):
    
        if use_flip:
            num_groups =  8
        else:
            num_groups =  4
    
        if agg_fun is None:
            out_channels = out_channels // num_groups

        super(IsoConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size, **kwargs)
        self.use_flip = use_flip
        self.num_groups = num_groups
        trans_params = [
            {'T':False,'Xflip':False,'Yflip':False}, #original
            {'T':True,'Xflip':False,'Yflip':True}, #rot 90
            {'T':False,'Xflip':True,'Yflip':True}, #rot 180
            {'T':True,'Xflip':True,'Yflip':False}, #rot 270
        ]
        if use_flip:
            trans_params = trans_params + [
            {'T':True,'Xflip':False,'Yflip':False}, #transposed
            {'T':False,'Xflip':False,'Yflip':True},  # T & rot 90
            {'T':True,'Xflip':True,'Yflip':True},  # T & rot 180
            {'T':False,'Xflip':True,'Yflip':False},  # T & rot 270
            ]
        self.trans_params = trans_params
        self.agg_fun = agg_fun

    def get_tranformed_weights(self):
        a =  [self.get_tranformed_weight(trans_param) for trans_param in self.trans_params]
        return torch.cat(a,dim=0)
    def get_tranformed_weight(self,trans_param):
        w = self.weight
        flip_dims = []
        if trans_param['Xflip']:
            flip_dims.append(2)
        if trans_param['Yflip']:
            flip_dims.append(3)
        if len(flip_dims) >= 0:
            w = torch.flip(w,flip_dims)
        if trans_param['T']:
            w = w.permute(0,1,3,2,4)
        return w

    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)

       
        return out


class IsoConv3dDropout(IsoConv3d):
    def __init__(self,*args,
        dropout_p=0.5,
        **kwargs
    ):
        super(IsoConv3dDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            
            out = self.dropout(out)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)

       
        return out


class IsoSEConv3d(IsoConv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        use_SE=True,
        SE_type = 'channel',
        agg_fun = torch.max,
        **kwargs
    ):
        super(IsoSEConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            use_flip = use_flip,
            agg_fun = agg_fun,
            **kwargs
        )
        
        if use_SE:
            if agg_fun is None:
                SE_channels = out_channels
                #self.SE_module = SELayer3D(out_channels)
            else:
                SE_channels =  out_channels * self.num_groups
            if SE_type == 'channel':
                self.SE_module = GroupSELayer3D(SE_channels,groups=self.num_groups)
            else:
            
                self.SE_module = GroupnDSELayer3D(SE_channels,axis=2,
                    kernel_w=1,use_softmax=False,groups=self.num_groups,
                    bias=False,n_latent_layers=0)
            
        else:
            self.SE_module = nn.Sequential()
        
    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            out = self.SE_module(out)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)
            out = self.SE_module(out)

       
        return out

class Iso3Conv3d(nn.Conv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        agg_fun = torch.max,
        **kwargs
    ):
    
        if use_flip:
            num_groups =  16
        else:
            num_groups =  8
    
        if agg_fun is None:
            out_channels = out_channels // num_groups

        super(Iso3Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size, **kwargs)
        self.use_flip = use_flip
        self.num_groups = num_groups
        trans_params = [
            {'T':False, 'Xflip':False,  'Yflip':False,	'Zflip':False},
            {'T':True,  'Xflip':False,  'Yflip':True,	'Zflip':False},
            {'T':False, 'Xflip':True,   'Yflip':True,	'Zflip':False},
            {'T':True,  'Xflip':True,   'Yflip':False,	'Zflip':False},
            {'T':True,	'Xflip':False,	'Yflip':False,	'Zflip':True}, 
            {'T':False,	'Xflip':False,	'Yflip':True,	'Zflip':True}, 
            {'T':True,	'Xflip':True,	'Yflip':True,	'Zflip':True}, 
            {'T':False,	'Xflip':True,	'Yflip':False,	'Zflip':True}, 
        ]
        if use_flip:
            trans_params = trans_params + [
            {'T':False, 'Xflip':False,  'Yflip':False,	'Zflip':True}, 
            {'T':True,  'Xflip':False,  'Yflip':True,	'Zflip':True}, 
            {'T':False, 'Xflip':True,   'Yflip':True,	'Zflip':True}, 
            {'T':True,  'Xflip':True,   'Yflip':False,	'Zflip':True}, 
            {'T':True,	'Xflip':False,	'Yflip':False,	'Zflip':False},
            {'T':False,	'Xflip':False,	'Yflip':True,	'Zflip':False},
            {'T':True,	'Xflip':True,	'Yflip':True,	'Zflip':False},
            {'T':False,	'Xflip':True,	'Yflip':False,	'Zflip':False},
        ]

        self.trans_params = trans_params
        self.agg_fun = agg_fun

    def get_tranformed_weights(self):
        a =  [self.get_tranformed_weight(trans_param) for trans_param in self.trans_params]
        return torch.cat(a,dim=0)
    def get_tranformed_weight(self,trans_param):
        w = self.weight
        flip_dims = []
        if trans_param['Xflip']:
            flip_dims.append(2)
        if trans_param['Yflip']:
            flip_dims.append(3)
        if trans_param['Zflip']:
            flip_dims.append(4)
        if len(flip_dims) >= 0:
            w = torch.flip(w,flip_dims)
        if trans_param['T']:
            w = w.permute(0,1,3,2,4)
        return w

    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)

       
        return out

class Iso3Conv3dDropout(Iso3Conv3d):
    def __init__(self,*args,
        dropout_p=0.5,
        **kwargs
    ):
        super(Iso3Conv3dDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            out = self.dropout(out)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)

       
        return out



class Iso3SEConv3d(Iso3Conv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        use_SE=True,
        SE_type='channel',
        agg_fun = torch.max,
        **kwargs
    ):
        super(Iso3SEConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            use_flip = use_flip,
            agg_fun = agg_fun,
            **kwargs
        )
        if use_SE:
            if agg_fun is None:
                SE_channels = out_channels
                #self.SE_module = SELayer3D(out_channels)
            else:
                SE_channels =  out_channels * self.num_groups
            if SE_type == 'channel':
                self.SE_module = GroupSELayer3D(SE_channels,groups=self.num_groups)
            else:
            
                self.SE_module = GroupnDSELayer3D(SE_channels,axis=2,
                    kernel_w=1,use_softmax=False,groups=self.num_groups,
                    bias=False,n_latent_layers=0)
            
        else:
            self.SE_module = nn.Sequential()
        
    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        if self.agg_fun is not None:
            weight = self.get_tranformed_weights()
            #bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight,bias=None)
            out = self.SE_module(out)
            out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
            out = self.agg_fun(out,dim=1)
            if isinstance(out,tuple):
                out = out[0]
            out = out + self.bias.view(1,-1,1,1,1)
        else:
            weight = self.get_tranformed_weights()
            bias = self.bias.repeat(self.num_groups)
            out = self._conv_forward(input, weight, bias)
            out = self.SE_module(out)

       
        return out



class IsoConv3d_Attagg(IsoConv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        agg_fun = torch.sum,
        dropout_p=0.5,
        **kwargs
    ):
        super(IsoConv3d_Attagg, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            use_flip = use_flip,
            agg_fun = agg_fun,
            **kwargs)
        self.ch_att = GroupChannelAttLayer3D(self.num_groups*out_channels,no_sigmoid=True,groups=self.num_groups)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        """ weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight, bias) """
        #if self.agg_fun is not None:
        weight = self.get_tranformed_weights()
        bias = self.bias.repeat(self.num_groups)
        out = self._conv_forward(input, weight,bias=bias)
        att = self.ch_att(out).view(out.shape[0],self.num_groups,-1,1,1,1)
        out = out.view(out.shape[0],self.num_groups,-1,*out.shape[2:])
        att = torch.softmax(att,dim=1)
        att = self.dropout(att)
        out = out * att
        out = self.agg_fun(out,dim=1)
        if isinstance(out,tuple):
            out = out[0]
        #out = out + self.bias.view(1,-1,1,1,1)
        return out
class IsoSepConv3d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        #agg_fun = torch.max,
        **kwargs
    ):
    
        super(IsoSepConv3d, self).__init__()
        if use_flip:
            num_groups =  8
        else:
            num_groups =  4
        kwargs.update({'agg_fun': None})

        self.conv = nn.Sequential(
            IsoConv3d(
                in_channels,
                out_channels*num_groups,
                kernel_size,
                use_flip = use_flip,
                #agg_fun = None,
                **kwargs),
            #nn.LeakyReLU(),
            #nn.BatchNorm3d(out_channels*num_groups),
            #nn.Conv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
            #SigmoidConv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
            SoftMaxConv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
        ) 
        
    def forward(self, x):
        return self.conv(x)

class Iso3SepConv3d(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        #agg_fun = torch.max,
        **kwargs
    ):
    
        super(Iso3SepConv3d, self).__init__()
        if use_flip:
            num_groups =  16
        else:
            num_groups =  8

        kwargs.update({'agg_fun': None})
        self.conv = nn.Sequential(
            Iso3Conv3d(
                in_channels,
                out_channels*num_groups,
                kernel_size,
                use_flip = use_flip,
                #agg_fun = None,
                **kwargs),
            #nn.LeakyReLU(),
            #nn.BatchNorm3d(out_channels*num_groups),
            #nn.Conv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
            #SigmoidConv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
            SoftMaxConv3d( out_channels*num_groups, out_channels, 1,groups=out_channels)
        ) 
        
    def forward(self, x):
        return self.conv(x)
            


       

class SoftMaxConv3d(nn.Conv3d):
    '''
    def __init__(self,*args):
        super().__init__(*args)        
    '''
    def forward(self, input):
        W = F.softmax(self.weight,dim=1)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            W, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)




class SigmoidConv3d(nn.Conv3d):
    '''
    def __init__(self,*args):
        super().__init__(*args)        
    '''
    def forward(self, input):
        W = F.sigmoid(self.weight)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            W, self.bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, W, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


 
class SpatialIsoSELayer3D(nn.Module):
    def __init__(self,channel,
        reduction=2,
        no_sigmoid=False,kernel_w=3,
        axis=2,dropout_p=0,
        bias=True,
        n_latent_layers=2,
        conv_fun = IsoConv3d,
        use_flip=True,
        ):
        super(SpatialIsoSELayer3D, self).__init__()
        if axis == None:
            pool_w = [None,None,None]
        else:
            pool_w = [1,1,1]
            pool_w[axis]= None
        pool_w = tuple(pool_w)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        #pad_size = tuple([math.floor(kernel_w)/2,math.floor(kernel_w)/2,1])

        layers = [
                nn.Conv3d(channel, channel // reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
        for i in range(n_latent_layers): 
            layers += [
                conv_fun(channel // reduction, channel // reduction,kernel_size=kernel_w, padding='same',use_flip=use_flip),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p)]
               
        layers.append(nn.Conv3d(channel // reduction, 1, kernel_size=1, bias=bias))
        if not no_sigmoid:
            layers.append(nn.Sigmoid())
        self.att = nn.Sequential(*layers)
        
    def get_att(self, x,mask=None):
        y = self.att(x)
        
        y = self.avg_pool(x)
        if mask is None:
            return y
        else:
            return y * mask

    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
class SpatialIsoAttLayer3D(SpatialSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)
        


class nDIsoSELayer3D(nn.Module):
    def __init__(self, channel, reduction=2,
        kernel_w=3,axis=None,dropout_p=0,use_softmax=False,
        bias=True,
        conv_fun = IsoConv3d,
        use_flip=True,
        n_latent_layers=2):
        super(nDIsoSELayer3D, self).__init__()        
        self.use_softmax = use_softmax
        self.axis = axis
        self.ch_att_layer = ChannelAttLayer3D(channel,reduction,no_sigmoid=True,dropout_p=dropout_p)
        self.sp_att_layer = SpatialIsoSELayer3D(
            channel,reduction,no_sigmoid=True,kernel_w=kernel_w,axis=axis,dropout_p=dropout_p,
            bias=bias,n_latent_layers=n_latent_layers,
            conv_fun = conv_fun,
            use_flip=use_flip,
            )

    def get_att(self, x,mask=None):
        ch_att =  self.ch_att_layer(x)
        sp_att =  self.sp_att_layer(x)
        y = ch_att + sp_att
        #Nonlinear actication (sigmoid or softmax)
        if self.use_softmax:
            if self.axis is not None: #softmax over the specified axis
                y = F.softmax(y,dim=2+ self.axis)
                
            else:#softmax over all axes
                y = F.softmax(y.view(y.shape[:2],-1),dim=-1).view(y.shape)

        else:
            y = torch.sigmoid(y)

        if mask is not None:
            y = y * mask
        return y
        
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
        

class nDIsoAttLayer3D(nDSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)


'''


class SpatialSELayer3D(nn.Module):
    def __init__(self, channel, reduction=2,axis=2,no_sigmoid=False):
        super(SpatialSELayer3D, self).__init__()
        self.axis = axis
        pool_w = [1,1,1]
        pool_w[axis]= None
        pool_w = tuple(pool_w)
        self.avg_pool = nn.AdaptiveAvgPool3d(pool_w)
        if not no_sigmoid:
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, 1,kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        else:  
            self.att = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, 1,kernel_size=1, bias=False)
            )

    def get_att(self, x,mask=None):
        y = self.avg_pool(x)
        y = self.att(y)
        #view_shape = [-1,1,1,1,1]
        #view_shape[self.axis+2] = self.input_size[self.axis]
        #y = y.view(*view_shape)
        if mask is None:
            return y
        else:
            return y * mask
    def forward(self, x,mask=None):
        return x * self.get_att(x,mask)
        

class SpatialAttLayer3D(SpatialSELayer3D):
    def forward(self, x,mask=None):
        return self.get_att(x,mask)
        '''