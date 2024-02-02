# MADI
MaDi contain model for multimodal diffusion for RS
This project is extended from Diffrec
> [Diffusion Recommender Model](https://arxiv.org/abs/2304.04971)
> 
> Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua

Dataset (Amazon-baby) is available at this [link](https://arxiv.org/abs/2304.04971)




### DiffRec contains models with out Pretrained Item Embedding
Move to DiffRec directory
```
cd ./DiffRec
```
then run model
```
# L-DiffRec
python  LDiffRec.py  --cuda --dataset=baby --data_path=../datasets/baby/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr1=0.001 --mean_type=x0

# DiffRec
python  DiffREC.py  --cuda --dataset=baby --data_path=../datasets/baby/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr=0.001 --mean_type=x0

# L-DiffRec with AE
python  LDiffRecAE.py  --cuda --dataset=baby --data_path=../datasets/baby/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr1=0.001 --mean_type=eps
```
```
# L-DiffRec with AE
python  LDiffRec.py  --cuda --dataset=baby --data_path=../datasets/baby/ --batch_size=400  --emb_size=10  --log_name=log --round=1 --lr1=0.001 --mean_type=x0
```

```
# MVAE
python MVAE.py --cuda

# lightGCN
python lightGCN.py
```


#### Make sure model have the compatible input files

Convert Amazon .json file to txt
```
python convert2LGCN.py
```

Convert Amazon .json file to npy
```
python convertAmzDataset.py
```

### Latent_Embedding contains models Using Pretrained Item Embedding
... updating
