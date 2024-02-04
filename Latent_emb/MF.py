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
from MF_helper.DNN import DNN
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
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
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
train_dataset = data_utils.DataMF(torch.FloatTensor(train_data.A), item_emb)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print('data ready.')
embSize = item_emb.shape[-1]
model = DNN(embSize, n_user).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

def evaluate(data_loader, data_te, mask_his, topN):
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    model.eval()
    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    emb = item_emb
    emb = emb.to(device)
    prediction = model(emb)
    prediction = torch.transpose(prediction, 0, 1)
    his_data = mask_his[e_idxlist]
    prediction[his_data.nonzero()] = -np.inf

    _, indices = torch.topk(prediction, topN[-1])
    indices = indices.cpu().numpy().tolist()
    predict_items.extend(indices)
    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results


best_recall, best_epoch = -100, 0
mask_train = train_data
lossFuncion = nn.MSELoss()
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break
    model.train()

    start_time = time.time()

    total_loss = 0.0
    batch = torch.FloatTensor(train_data.A)
    batch = batch.to(device)
    emb = item_emb
    emb = emb.to(device)
    optimizer.zero_grad()
    output = model(emb)
    output = torch.transpose(output, 0, 1)
    loss = lossFuncion(output, batch)
    loss.backward()
    total_loss += loss.item() 
    optimizer.step()
   
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, mask_train, eval(args.topN))
        test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

# print('==='*18)
# print("End. Best Epoch {:03d} ".format(best_epoch))
# evaluate_utils.print_results(None, best_results, best_test_results)   





# test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
# print('==='*18)
# evaluate_utils.print_results(None, test_results, test_results)   
# print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

