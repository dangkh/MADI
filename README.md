# MADI
This project is extended from Diffrec[link] 

DiffRec contain model for discrete Diffusion  for RS
python main.py  --dataset=baby --data_path=../datasets/baby/ --batch_size=400 --emb_size=5 --noise_scale=0.005 --mean_type=x0 --steps=10 --noise_min=0.005 --noise_max=0.01 --sampling_steps=5 --reweight=1 --log_name=log --round=1 --cuda
'''
1091
'''
Data Augmentation run
python main.py  --dataset=ml-1m_clean --data_path=../datasets/ml-1m_clean/ --batch_size=400 --emb_size=5 --noise_scale=0.005 --mean_type=x0 --steps=10 --noise_min=0.005 --noise_max=0.01 --sampling_steps=5 --reweight=1 --log_name=log --round=1 --cuda --mask_BCE

MaDi contain model for multimodal diffusion for RS