import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset


def removeData(dataList, numItem, numUser):
    newData =  dataList[np.where(dataList[:,1] < numItem)]
    newData =  newData[np.where(newData[:,0] < numUser)]
    return newData

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid

    for dataList in [valid_list, test_list]:
        for uid, iid in dataList:
            if uid > uid_max:
                uid_max = uid
            if iid > iid_max:
                iid_max = iid            
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data, embed):
        self.data = data
        self.embed = embed
        self.userEmb = []
        self.maxItem = len(embed)
        for ii in range(len(self.data)):
            self.userEmb.append(self.index2itemEm(data[ii]))
        self.pos = []
        for line in self.data:
            ll = torch.where(line == 1)[0]
            lenLL = len(ll)
            compensation = len(self.data[0]) - lenLL
            lcom = torch.zeros(compensation)
            newLL = torch.cat((ll, lcom), 0)
            self.pos.append([lenLL, newLL])

    def index2itemEm(self, itemIndx):
        output = []
        clickedItem = torch.where(itemIndx == 1)[0]
        index = torch.randperm(len(clickedItem))
        clickedItem = clickedItem[index]
        counter = 0
        for ii in clickedItem:
            output.append(self.embed[ii.item()])
            counter += 1
            if counter > (self.maxItem-1):
                break
        compensationNum = self.maxItem -counter
        compensationFeat = torch.zeros(compensationNum*64)
        output.append(compensationFeat)
        return torch.cat(output)

    def __getitem__(self, index):
        item = self.data[index]
        embed = self.userEmb[index]
        return item, embed, self.pos[index]

    def __len__(self):
        return len(self.data)