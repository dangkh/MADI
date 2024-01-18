from lightGCN_helper.main import LGCN

model = LGCN('baby')
res = model.train_getRes()
print(res)