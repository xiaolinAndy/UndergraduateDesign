'''
This file documents the experiment parimeters to run a model.
'''

def default_config():
	config = {}

	# The folder name to store all intermediary results.
	config['exp_folder'] = './experiment'
	# The name associated with all files for the experiment.
	config['file_prefix'] = ''

	## Define which pretraining method to use. 
	
	# Choose which model to get embeddings from.
	# Choices are 'vhred', 'dual_encoder', 'fair_autoencoder', 'tweet2vec'.
	config['pretraining'] = 'vhred'
	# True if we should reduce the dimensionality of the pretrained embeddings.
	config['use_pca'] = True
	config['pca_components'] = 50
	# True if we should oversample the data such that the candidate responses have the same distribution of scores across different lengths.
	config['oversample_length'] = True
	config['embedding_type'] = 'CONTEXT'
	config['vhred_prefix'] = '../vhred/1521439677.69_TiebaModel_VHRED'
	config['vhred_dict'] = '../vhred/dict.pkl'
	config['vhred_embeddings_file'] = ''
	config['vhred_dim'] = 1000

	## Model parameters.

	# True if the ADEM model should include the cMr term.
	config['use_c'] = True
	# True if the ADEM model should include the rNr' term.
	config['use_r'] = True
	# Regularization constants on the (M, N) parameters.
	config['l2_reg'] = 0.075#0.1#0.1#0.5
	config['l1_reg'] = 0.0
	config['bs'] = 32

	## Training parameters.

	config['max_epochs'] = 200
	config['val_percent'] = 0.15
	config['test_percent'] = 0.15
	# Whether to validate on 'rmse' or 'pearson' correlation.
	config['validation_metric'] = 'rmse'
	# If set to a model 'hred', 'de', 'tfidf', 'human', we will leave this model out of the training set.
	config['leave_model_out'] = None
	# data
	config['contexts'] = '../tieba/final.tieba.contexts.txt'
	config['true_responses'] = '../tieba/final.true.responses.txt'
	config['tfidf'] = '../tieba/final.tfidf.responses.txt'
	config['de'] = '../tieba/final.dual_encoder.responses.txt'
	config['VHRED'] = '../tieba/final.VHRED.responses.txt'
	config['human'] = '../tieba/final.human.responses.txt'
	config['score'] = '../data/statistics.pkl'
	config['index'] = '../data/index'

	return config
