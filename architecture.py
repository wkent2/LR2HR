import os

import typing
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import pytorch_lightning as pl
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint,ModelSummary

from data import *


def get_conv_dim(in_dim,layer_params):

    H_in, W_in, D_in = in_dim[0], in_dim[1], in_dim[2]

    name,nfilter,k,s,p = layer_params

    H_out = ((H_in+2*p-k)/s)+1
    W_out = ((W_in+2*p-k)/s)+1
    D_out = ((D_in+2*p-k)/s)+1

    return (int(H_out),int(W_out),int(D_out))

def get_pool_dim(in_dim,layer_params):

    H_in, W_in, D_in = in_dim[0], in_dim[1], in_dim[2]

    name,nfilter,k,s,p = layer_params

    H_out = ((H_in+2*p-k)/s)+1
    W_out = ((W_in+2*p-k)/s)+1
    D_out = ((D_in+2*p-k)/s)+1

    return (int(H_out),int(W_out),int(D_out))


def get_convolution_dims(layer_params):
    '''
    Adds input and ouput dimensions for each convolution
    '''
    first = True
    for i in range(len(layer_params)):
        if layer_params[i][0] == "Conv3D":
            # Check first 
            if first:
                layer_params[i][1] = [1,layer_params[i][1]]
                first = False
            else:
                # Find previous Conv2d layer
                index = i - 1
                while layer_params[index][0] != "Conv3D":
                    index -=1
                # Set input channels to be output channels of last Conv2d layer
                layer_params[i][1] = [layer_params[index][1][1],layer_params[i][1]]
    return layer_params

def get_linear_size(layer_params,imsize):
    '''
    Calculates the flattened inputs size for first linear layer
    '''
    # Find params for all preceding layers
    
    index = 0
    # Find the first linear layer
    for i in range(len(layer_params)):
        if layer_params[i][0]=='Dense':
            index=i
            break

    # Extract image sizes
    _,H,W,D= imsize
    print("Input Data Dimensions:",H,W,D)

    # Iterate through layers to get dimensions
    for i in range(index):
        l_type = layer_params[i][0]
        if  l_type == 'Conv3D':
            H,W,D = get_conv_dim((H,W,D),layer_params[i])
        elif (l_type == 'AvgPool3D') or (l_type == 'MaxPool3D'):
            H,W,D = get_pool_dim((H,W,D),layer_params[i])

   
    # Comput linear dimension (heigh*width*n_channel)
    linear_size = H*W*D*layer_params[index-2][1][1]
    print("Dense layer value",layer_params[index-2][1][1])

    return linear_size

def parse_arch(arch_string):
    '''
    Parses input archictecture string into layer parameters
    '''
    arch_string = arch_string.split(',')
    layer_params = []
    names = []
    for i in range(len(arch_string)):
        layer_params.append(parse_string(arch_string[i]))
        names.append(str(i))


    layer_params =  get_convolution_dims(layer_params)
    print(layer_params)

    return layer_params,names

def get_int(string,index):
    '''
    Extracts integer number from string.
    Starts looking at index
    '''

    while index < len(string) and string[index].isdigit()  :
        index += 1
    num = int(string[1:index])

    return num,index

def parse_string(string):
    '''
    This function takes in a string and determines what architecture layer
    it is coding for and extracts necessary parameters.

    Types;
    'c' : Conv2d layer
    
    
    '''

    # Check layer type
    if string[0] == 'c':
        layer = "Conv3D"
    elif string[0] == 'a':
        layer = "AvgPool3D"
    elif string[0] == 'm':
        layer = "MaxPool3D"
    elif string[0] == 'l' or string[0] == 'd':
        layer = "Dense"

    nfilter, index = get_int(string,1)


    options = string[index:]
    # Kernel size
    if 'k' in options:
        k = int(options[options.index('k')+1])
    else:
        k=3
    #Let's see if stride was specified
    if 's' in options:
        s = int(options[options.index('s')+1])
    else:
        s=1
    #Let's see if padding was specified
    if 'p' in options:
        p = options[options.index('k')+1]
    else:
        p = 1

    return [layer,nfilter,k,s,p]

class CNNModel(nn.Module):
    def __init__(self, img_shape,out_shape, arcstr,reg=None):
        super(CNNModel, self).__init__()

        # Extract layer parameters from arch. string
        layer_params, names = parse_arch(arcstr)
        
        self.features = nn.Sequential()
        flattened_yet = False

        first_linear = get_linear_size(layer_params,img_shape)
        
        for i in range(len(layer_params)):
            layertype,nfilter,k,s,p = layer_params[i]
            if layertype == "Conv3D":
                self.features.add_module(f'conv3d_{len(self.features)}', nn.Conv3d(nfilter[0],nfilter[1], (k, k, k), stride=(s, s, s), padding=p))
                self.features.add_module(f'leaky_relu_{len(self.features)}', nn.LeakyReLU(0.1))
            elif layertype == "AvgPool3D":
                self.features.add_module(f'avg_pool3d_{len(self.features)}', nn.AvgPool3d(kernel_size=(k, k, k), stride=(s, s, s),padding=p))
            elif layertype == "MaxPool3D":
                self.features.add_module(f'max_pool3d_{len(self.features)}', nn.MaxPool3d(kernel_size=(k, k, k), stride=(s, s, s),padding=p))
            elif layertype == "Dense":
                if not flattened_yet:
                    self.features.add_module(f'flatten_{len(self.features)}', nn.Flatten())
                    self.features.add_module(f'linear_{len(self.features)}', nn.Linear(first_linear,layer_params[i][1]))
                    self.features.add_module(f'relu_{len(self.features)}', nn.ReLU())
                    flattened_yet = True
                else:
                    self.features.add_module(f'linear_{len(self.features)}', nn.Linear(layer_params[i-1][1],layer_params[i][1]))
                    self.features.add_module(f'relu_{len(self.features)}', nn.ReLU())

        self.features.add_module(f'linear_{len(self.features)}', nn.Linear(layer_params[i][1],out_shape))
        
        if reg is not None:
            for module in self.features.modules():
                if isinstance(module, typing.Union[nn.Linear,nn.LazyConv3d,nn.Conv3d,nn.LazyLinear]):
                    if reg == 'l1':
                        module.weight.regularizer = nn.L1Loss()
                    elif reg == 'l2':
                        module.weight.regularizer = nn.MSELoss()
                    elif reg == 'l1_l2':
                        module.weight.regularizer = nn.L1Loss() + nn.MSELoss()
                    else:
                        pass

    def forward(self, x):
        x = self.features(x)
        
        x[:, :3] = nn.Softmax(dim=1)(x[:, :3])

        return x

    def parse_arcstring(self, f):
        if not isinstance(f, str):
            return (int(f), 3, 1, "same", "Conv3D")
        f_orig = f
        f = f.lower()
        
        if f[0].isdigit():
            layertype = "Conv3D"
        elif f[0] == "c":
            layertype = "Conv3D"
            f = f[1:]
        elif f[0] == "d":
            layertype = "Dense"
            f = f[1:]
        elif f[0] == "a":
            layertype = "AvgPool3D"
            f = f[1:]
        elif f[0] == "m":
            layertype = "MaxPool3D"
            f = f[1:]
        else:
            raise ValueError(f"Improper architecture specification {f_orig} - failed at layertype parse.")
        
        firstletter = None
        for i, c in enumerate(list(f)):
            if not c.isdigit():
                firstletter = i
                break
        
        if firstletter is None:
            return (int(f), 3, 1, "same", layertype)
        
        n_filters = int(f[:firstletter])
        options = f[firstletter:]

        if "k" in options:
            k = int(options[options.index("k") + 1])
        else:
            k = 3
        
        if "s" in options:
            s = int(options[options.index("s") + 1])
        else:
            s = 1
        
        if "p" in options:
            p = options[options.index("p") + 1]
        else:
            p = "a"
        
        if p == "a":
            p = "same"
            s = 1 if layertype == 'Conv3D' else s
        elif p == "v":
            p = "valid"
        else:
            raise ValueError(f"Improper padding specified, {p} is not a proper value. Arcstring was {f_orig}")

        
        return (n_filters, k, s, p, layertype)


