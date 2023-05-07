

from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single
from numpy import isin
import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class SymGroupConv3d(nn.Conv3d): # only rotate X & Y
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False, 
        agg_fun = torch.max,
        **kwargs
    ):
        '''
        additional params:
            use_flip: if use mirroring or not. default: False
            agg_fun: aggregation function. (Options:
                None:  perform 4 * (conv(out_channels / 4)), no aggregation 
                torch.mean: perform 4 * (conv(out_channels)) -> mean pooling to out_channels
                torch.max: perform 4 * (conv(out_channels)) -> max pooling to out_channels
        '''
    
        if use_flip:
            num_groups =  8
        else:
            num_groups =  4
    
        if agg_fun is None:
            out_channels = out_channels // num_groups

        super(SymGroupConv3d, self).__init__(
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

class Iso3Conv3d(nn.Conv3d):  #  rotate X & Y (0,90,180,270 degree) & Z(0,180 degree)
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


""" 

class SymGroupConv3dDropout(SymGroupConv3d):
    def __init__(self,*args,
        dropout_p=0.5,
        **kwargs
    ):
        super(SymGroupConv3dDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
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


class IsoSEConv3d(SymGroupConv3d):
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



class Iso3Conv3dDropout(Iso3Conv3d):
    def __init__(self,*args,
        dropout_p=0.5,
        **kwargs
    ):
        super(Iso3Conv3dDropout, self).__init__(*args, **kwargs)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
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

class SymGroupConv3d_Attagg(SymGroupConv3d):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        use_flip = False,
        agg_fun = torch.sum,
        dropout_p=0.5,
        **kwargs
    ):
        super(SymGroupConv3d_Attagg, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            use_flip = use_flip,
            agg_fun = agg_fun,
            **kwargs)
        self.ch_att = GroupChannelAttLayer3D(self.num_groups*out_channels,no_sigmoid=True,groups=self.num_groups)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
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
            SymGroupConv3d(
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
            


       
 """