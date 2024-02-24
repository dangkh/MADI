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


# class MF_old(nn.Module):
#     """docstring for MF_old"""
#     def __init__(self, factor_num = 16, num_users = None, num_items = None):
#         super(MF_old, self).__init__()
#         self.num_factors = num_factors
#         self.num_users = num_users
#         self.num_items = num_items
#         self.P = nn.Embedding(num_users, num_factors)
#         self.Q = nn.Embedding(num_items, num_factors)
#         self.user_bias = nn.Embedding(num_users, 1)
#         self.item_bias = nn.Embedding(num_items, 1)
#         self.affine_output = nn.Linear(in_features=num_factors, out_features=1)
#         self.logistic = nn.Sigmoid()

#     def forward(self, user_id, item_id):
#         user_embedding = self.P(user_id)
#         item_embedding = self.Q(item_id)
#         element_product = torch.mul(user_embedding, item_embedding)
#         logits = self.affine_output(element_product)
#         rating = self.logistic(logits)
#         return rating
        
layers = [64,32,16,8]
dropout = 0.2

class MF_old(nn.Module):
    def __init__(self, factor_num = 16, num_users = None, num_items = None):
        super(MF_old, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num_mf = factor_num
        self.factor_num_mlp =  layers[0] / 2
        self.layers = layers
        self.dropout = dropout

        self.embedding_user_mlp = nn.Embedding(num_users, factor_num)
        self.embedding_item_mlp = nn.Embedding(num_items, factor_num)

        self.embedding_user_mf = nn.Embedding(num_users, factor_num)
        self.embedding_item_mf = nn.Embedding(num_items, factor_num)

        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            if idx == 0:
                self.fc_layers.append(torch.nn.Linear(factor_num * 2, out_size))
            else:
                self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.affine_output = nn.Linear(in_features=layers[-1] + self.factor_num_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)

        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.affine_output.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating.squeeze()