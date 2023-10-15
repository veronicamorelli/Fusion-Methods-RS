import pickle
import os

def load_dict(path):
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

def save_list_results(results_path, name_list, data):
	if not(os.path.exists(results_path)):
		os.mkdir(results_path)
	
	with open(results_path + name_list, 'w') as fp:
		for item in data:
			fp.write(str(item) + "\n")
