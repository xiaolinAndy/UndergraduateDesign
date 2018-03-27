'''
This file is the training pipeline for the ADEM model.
'''
import argparse
from experiments import *
import os
import cPickle
from models import ADEM
from preprocess import load_data

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='default_config')
	args = parser.parse_args()
	return args

def create_experiment(config):
	if not os.path.exists(config['exp_folder']):
		os.makedirs(config['exp_folder'])

if __name__ == "__main__":
	args = parse_args()
	config = eval(args.prototype)()
	print 'Beginning...'
	# Set up experiment directory.
	create_experiment(config)
	# This will load our training data.
	print 'Loading data...'
	data = load_data(config)

	# Train our model.
	adem = ADEM(config)
	print 'Training...'
	adem.train_eval(data, use_saved_embeddings=True)
	print 'Trained!'
	adem.save()


	
