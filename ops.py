import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import ipdb




def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape
    y_shapes = y.shape
    y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

    return torch.cat((x, y2),1)

def conv_prev_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape
    y_shapes = y.shape
    if x_shapes[2:] == y_shapes[2:]:
        y2 = y.expand(x_shapes[0],y_shapes[1],x_shapes[2],x_shapes[3])

        return torch.cat((x, y2),1)

    else:
        print(x_shapes[2:])
        print(y_shapes[2:])



def batch_norm_1d(x):
    x_shape = x.shape[1]
    batch_nor = nn.BatchNorm1d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    batch_nor = batch_nor.cuda()

    output = batch_nor(x)
    return output


def batch_norm_1d_cpu(x):
    x_shape = x.shape[1]
    # ipdb.set_trace()
    # batch_nor = nn.BatchNorm1d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    # output = batch_nor(x)
    output = x
    return output





def batch_norm_2d(x):
    x_shape = x.shape[1]
    batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    batch_nor = batch_nor.cuda()
    output = batch_nor(x)
    return output


def batch_norm_2d_cpu(x):
    # x_shape = x.shape[1]
    # batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    # batch_nor = batch_nor
    # output = batch_nor(x)
    output = x
    return output



def sigmoid_cross_entropy_with_logits(inputs,labels):
    loss = nn.BCEWithLogitsLoss()
    output = loss(inputs, labels)
    return output



def reduce_mean(x):
    output = torch.mean(x,0, keepdim = False)
    output = torch.mean(output,-1, keepdim = False)
    return output


def reduce_mean_0(x):
    output = torch.mean(x,0, keepdim = False)
    return output


def l2_loss(x,y):
    loss_ = nn.MSELoss(reduction='sum')
    l2_loss_ = loss_(x, y)/2
    return l2_loss_



def lrelu(x, leak=0.2):
    z = torch.mul(x,leak)
    return torch.max(x, z)







