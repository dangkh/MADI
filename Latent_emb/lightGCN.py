"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='baby', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/baby/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'
emb_path = args.data_path +  'iEmb.npy'
item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))

emb_path = args.data_path +  'uEmb.npy'
user_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))
resMap = torch.matmul(user_emb, torch.transpose(item_emb, 0, 1))

print(f'{user_emb.shape} user embedding shape, {item_emb.shape} item embedding shape')

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataSimple(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print('data ready.')


def evaluate(data_loader, data_te, mask_his, topN):
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    for batch_idx, batchInfo in enumerate(data_loader):
        index, batch = batchInfo
        # infer using original data

        tmp = torch.matmul(user_emb, torch.transpose(item_emb, 0, 1))
        prediction = tmp[index]

        his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
        prediction[his_data.nonzero()] = -np.inf

        _, indices = torch.topk(prediction, topN[-1])
        indices = indices.cpu().numpy().tolist()
        predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))

print('==='*18)
evaluate_utils.print_results(None, test_results, test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





