import json
import numpy as np
'''
this file convert file .json to LightGCN
'''
listF  = ['train', 'test', 'val']
for idx, file in enumerate(listF):
	f = open(f'./datasets/baby/{file}.json')
	sourceFile = open(f'./datasets/baby/{listF[idx]}.txt', 'w')
	# returns JSON object as 
	# a dictionary
	data = json.load(f)
	for user in data:
		listUI = data[user]
		print(user, *listUI, file = sourceFile)

	sourceFile.close()
	f.close()