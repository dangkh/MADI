import world
import utils
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

Neg_k = 1
w = None
for epoch in range(world.TRAIN_epochs):
    start = time.time()
    if epoch %10 == 0:
        cprint("[TEST]")
        Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
