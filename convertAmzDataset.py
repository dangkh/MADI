import json
import numpy as np

listF  = ['train', 'test', 'val']
listOf  = ['train_list', 'test_list', 'valid_list']
for idx, file in enumerate(listF):
	f = open(f'./datasets/baby/5-core/{file}.json')
	 
	# returns JSON object as 
	# a dictionary
	data = json.load(f)
	r = []
	for user in data:
		listUI = data[user]
		for item in listUI:
			r.append([int(user), int(item)])
	r = np.asarray(r)
	np.save(f'./datasets/baby/5-core/{listOf[idx]}.npy', r)

	# Closing file
	f.close()