import lightGCN_helper.world as world
import lightGCN_helper.dataloader as dataloader
import lightGCN_helper.model as model
import lightGCN_helper.utils as utils
from pprint import pprint
import os
path = os.getcwd()
path = path.split('\\')
mypath = ''
for xx in path[:-1]:
    mypath = mypath + f'{xx}\\'
if world.dataset in ['baby', 'sport', 'cloth']:
    dataset = dataloader.Loader(path=f"{mypath}datasets\\"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}