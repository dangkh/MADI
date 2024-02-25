"""
Train a MVAE model for recommendation
using settings in diffrec
"""

import argparse
from ast import parse
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from VAE_helper.Autoencoder import AutoEncoder as AE
from VAE_helper.Autoencoder import loss_function
import evaluate_utils
import data_utils

import random
random_seed = 1001
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='baby', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='../datasets/baby/', help='load data path')
parser.add_argument('--emb_path', type=str, default='../datasets/baby/')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Autoencoder')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay for Autoencoder')
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')

# params for the Autoencoder
parser.add_argument('--in_dims', type=str, default='[64]', help='the dims for the encoder')
parser.add_argument('--reduce_dims', type=int, default='2', help='the dims reduce before input the encoder')
parser.add_argument('--act_func', type=str, default='tanh', help='activation function for autoencoder')
parser.add_argument('--lamda', type=float, default=0.05, help='hyper-parameter of multinomial log-likelihood for AE: 0.01, 0.02, 0.03, 0.05')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for AE: Adam, AdamW, SGD, Adagrad, Momentum')
parser.add_argument('--anneal_cap', type=float, default=0.005)
parser.add_argument('--anneal_steps', type=int, default=500)
parser.add_argument('--vae_anneal_cap', type=float, default=0.3)
parser.add_argument('--vae_anneal_steps', type=int, default=200)
parser.add_argument('--reparam', type=bool, default=True, help="Autoencoder with variational inference or not")

# params for diffusion

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")
# device = "cpu"
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'
emb_path = args.emb_path +  'iEmb.npy'
item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A), item_emb)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

mask_tv = train_data + valid_y_data

print('data ready.')
### Build Autoencoder ###

in_dims = eval(args.in_dims)
# get shape of item embedding:
embDim = item_emb.shape[-1]
in_dims.append(args.reduce_dims * n_item)
Autoencoder = AE(in_dims, conDim = embDim, shrinkSize = args.reduce_dims).to(device)
param_num = 0
AE_num = sum([param.nelement() for param in Autoencoder.parameters()])
print("Number of parameters:", AE_num)
if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(
        Autoencoder.parameters(), lr=args.lr, initial_accumulator_value=1e-8, weight_decay=args.wd)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(Autoencoder.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'AdamW':
    optimizer = optim.AdamW(Autoencoder.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(Autoencoder.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(Autoencoder.parameters(), lr=args.lr, momentum=0.95, weight_decay=args.wd)

print("models ready.")


def evaluate(data_loader, data_te, mask_his, topN):
    Autoencoder.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, (batch, emb) in enumerate(train_loader):
            batch = batch.to(device)
            emb = emb.to(device)

            # mask map
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]

            batch_recon, z, mu, logvar, prediction = Autoencoder(emb)
            prediction[his_data.nonzero()] = -np.inf  # mask ui pairs in train & validation set

            _, indices = torch.topk(prediction, topN[-1])  # topk category idx

            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
update_count = 0
update_count_vae = 0

mask_train = train_data
recLoss = nn.BCEWithLogitsLoss()

listSFBatch = []
for epoch in range(0, args.epochs):
    if epoch - best_epoch >= 200:
        print('-'*18)
        print('Exiting from training early')
        break

    Autoencoder.train()

    start_time = time.time()

    total_loss = 0.0
    for batch_idx, (batch, emb) in enumerate(train_loader):
        batch = batch.to(device)
        emb = emb.to(device)
        optimizer.zero_grad()
        
        batch_recon, z, mu, logvar, pred = Autoencoder(emb)
        loss1 = loss_function(batch_recon, emb, mu, logvar, z)
        loss2 = recLoss(pred, batch)

        loss = loss1 + loss2

        loss.backward()
        total_loss += loss.item() 
        optimizer.step()

    update_count += 1
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, mask_train, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall: # recall@20 as selection
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results

    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))



