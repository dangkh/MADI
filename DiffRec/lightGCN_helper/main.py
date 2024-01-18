import lightGCN_helper.world as world
import lightGCN_helper.utils as utils
from lightGCN_helper.world import cprint
import torch
import numpy as np
import time
import lightGCN_helper.Procedure as Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import lightGCN_helper.register as register
from lightGCN_helper.register import dataset

class LGCN(object):
	"""docstring for LGCN"""
	def __init__(self, dataname):
		super(LGCN, self).__init__()
		self.dataname = dataname

	def train_getRes(self):
		Recmodel = register.MODELS[world.model_name](world.config, dataset)
		Recmodel = Recmodel.to(world.device)
		bpr = utils.BPRLoss(Recmodel, world.config)

		Neg_k = 1
		w = None
		results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
		for epoch in range(world.TRAIN_epochs):
		    start = time.time()
		    if epoch %10 == 0:
		        cprint("[TEST]")
		        results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
		    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
		    print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
		return results