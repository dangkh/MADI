import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_

class DNN(nn.Module):
    """
    A deep neural network for the reverse process of latent diffusion.
    """
    def __init__(self, in_dims, out_dims, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.linear = nn.Linear(in_dims, out_dims)
        self.dropout = nn.Dropout(dropout)
        self.act_func = nn.Sigmoid()
        self.apply(xavier_normal_initialization)
    
    def forward(self, x):
        return self.linear(x)


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)    


class MF_old(nn.Module):
    """docstring for MF_old"""
    def __init__(self, num_factors = 16, num_users = None, num_items = None):
        super(MF_old, self).__init__()
        self.num_factors = num_factors
        self.num_users = num_users
        self.num_items = num_items
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        # self.user_bias = nn.Embedding(num_users, 1)
        # self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        Q_i = torch.transpose(Q_i, 0, 1)
        # b_u = self.user_bias(user_id)
        # b_i = self.item_bias(item_id)
        # print(P_u.shape, Q_i.shape)
        # stop
        # outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        outputs = torch.matmul(P_u, Q_i)
        return outputs
        

class GMF(object):
    """docstring for GMF"""
    def __init__(self, arg):
        super(GMF, self).__init__()
        self.arg = arg
        

class MLP(object):
    """docstring for MLP"""
    def __init__(self, arg):
        super(MLP, self).__init__()
        self.arg = arg
                        
class NMF(object):
    """docstring for NMF"""
    def __init__(self, arg):
        super(NMF, self).__init__()
        self.arg = arg
