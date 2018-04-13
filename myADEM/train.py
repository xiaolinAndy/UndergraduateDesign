'''
This file is the training pipeline for the ADEM model.
'''
import argparse
import lasagne
from experiments import *
import os
import cPickle
from models import ADEM
from preprocess import load_data, load_test_data
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.insert (0, '../embedding')
from ppdb_word_model import *
from ppdb_lstm_model import *
from ppdb_utils import train, getWordmap, getPPDBData, getPPDBData_unsupervise
from params import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='default_config')
	parser.add_argument("--model", type=str, help="use supervised training or unsupervised", default='supervised_simple')
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
	if os.path.exists(config['exp_folder'] + '/dataset.pkl'):
		print 'Loading from existed data...'
		data = cPickle.load(open(config['exp_folder'] + '/dataset.pkl', 'rb'))
	else:
		print 'Loading from new data...'
		data = load_data(config)

	# Train our model.
	if args.model == 'supervised_simple':
		adem = ADEM(config)
		print 'Training...'
		adem.train_eval(data, config, use_saved_embeddings=False)
		print 'Trained!'
		adem.save()
	else:
		params = params()
		params.LW = config['LW']
		params.LC = config['LC']
		params.outfile = config['outfile']
		params.batchsize = config['batchsize']
		params.hiddensize = config['emb_dim']
		params.wordfile = config['word_vec']
		params.updatewords = config['updatewords']
		params.train = config['paraphrase']
		params.margin = config['margin']
		params.epochs = config['emb_epoch']
		params.learner = lasagne.updates.adam
		params.eta = config['eta']
		params.clip = config['clip']
		params.type = 'MIX'
		params.save = False
		(words, We) = getWordmap(config['word_vec'])
		if args.model == 'supervised':
			examples = getPPDBData (params.train, words)
			model = ppdb_word_model(We, params)
			adem = ADEM(config)
			print 'Training...'
			train(model, examples, words, params, adem, config, data, args.model)
		elif args.model == 'unsupervised':
			params.train = config['triple']
			examples = getPPDBData_unsupervise(params.train, words)
			model = unsupervised_adem_model(We, params)
			print 'Training...'
			adem = ADEM(config)
			train(model, examples, words, params, adem, config, data, args.model)
		elif args.model == 'unsupervised_complex':
			params.train = config['triple']
			examples = getPPDBData_unsupervise(params.train, words)
			model = unsupervised_adem_model_complex(We, params)
			print 'Training...'
			adem = ADEM(config)
			train(model, examples, words, params, adem, config, data, args.model)
		elif args.model == 'unsupervised_ruber':
			params.train = config['triple']
			examples = getPPDBData_unsupervise(params.train, words)
			model = unsupervised_ruber_model(We, params)
			print 'Training...'
			adem = ADEM(config)
			train(model, examples, words, params, adem, config, data, args.model)
		elif args.model == 'unsupervised_lstm':
			params.train = config['triple']
			params.layersize = 300
			params.peephole = True
			examples = getPPDBData_unsupervise(params.train, words)
			model = unsupervised_adem_lstm_model(We, params)
			print 'Training...'
			adem = ADEM(config)
			train(model, examples, words, params, adem, config, data, args.model)
		else:
			model = unsupervised_adem_model(We, params)
			params = cPickle.load(open(config['model_name'], 'rb'))
			model.We.set_value(params[0].get_value())
			model.M.set_value(params[1].get_value())
			model.N.set_value(params[2].get_value())
			print 'Training...'
			config['iclr_embs'] = '../embedding/unsupervised_embeddings.pkl'
			adem = ADEM(config)
			train(model, None, words, params, adem, config, data, args.model)






	#data = load_test_data(config)
	'''data = adem.pretrainer.get_embeddings(data)
	c = data[0]['c_emb']
	r_gt = data[0]['r_gt_emb']
	r_models = data[0]['r_model_embs']['tfidf']
	c = np.array(c, dtype = float)
	r_gt = np.array(r_gt, dtype=float)
	r_models = np.array(r_models, dtype=float)
	print len(c), len(r_gt)
	num = np.dot(c, r_gt)
	denom = np.linalg.norm(c) * np.linalg.norm(r_gt)
	sim_1 = 0.5 + 0.5 * num / denom
	num = np.dot(r_models, r_gt)
	denom = np.linalg.norm(r_models) * np.linalg.norm(r_gt)
	sim_2 = 0.5 + 0.5 * num / denom
	print sim_1, sim_2'''

	'''data = adem.pretrainer.get_embeddings_test(data)
	c = data[0]
	c = np.array(c, dtype=float)
	index, biggest_sim = 0, 0
	for i, r in enumerate(data[1:]):
		r = np.array(r, dtype=float)
		num = np.dot(c, r)
		denom = np.linalg.norm(c) * np.linalg.norm(r)
		sim = 0.5 + 0.5 * num / denom
		print i, sim
		if biggest_sim < sim:
			index = i
			biggest_sim = sim
	print index, biggest_sim'''



	
