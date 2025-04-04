import numpy as np
import pandas as pd
import yaml
import argparse
import os

def parseargs():
    p = argparse.ArgumentParser(description="Compiles the hyperparameters of all versions in a results directory")
    p.add_argument("-f","-folder", type=str, default='../resultsLR2HR/lightning_logs', help="Path to versions")
    p.add_argument("-s","-savename", type=str, default='./compiled_hparams.csv', help="Savename for haparams")

    args = p.parse_args()

    return args 

def load_hyperparameters(file_path):
    with open(file_path, "r") as file:
        hyperparams = yaml.safe_load(file)  # Load YAML content safely
    return hyperparams

def check_file_exists(filepath,err_msg):
    if not os.path.exists(filepath):
        print(err_msg)
        exit()

if __name__ == '__main__':

	args = parseargs()

	# Find all folders
	folders = [f for f in os.listdir(args.f) if f.startswith('version_')]
	folders = [f for f in folders if os.path.exists(os.path.join(args.f,f,'hparams.yaml'))]

	# load all hparams
	hparams = [load_hyperparameters(os.path.join(args.f,f,'hparams.yaml')) for f in folders ]

	# Figure out how many fields
	allkeys = np.concatenate([np.array(list(d.keys())) for d in hparams])
	keys = np.unique(allkeys)

	# Load fields into pandas df 
	indexes = [int(f.split('_')[1]) for f in folders]
	dictionary = {}
	for key in keys:
		dictionary[key] = len(indexes)*[np.nan]

	for i,h in enumerate(hparams):
		for key in h.keys():
			dictionary[key][i] = h[key]

	data = pd.DataFrame(dictionary,index=indexes)
	data = data.sort_index()
	data.to_csv(args.s)










