from pretrain import *
import os
import theano
import theano.tensor as T
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from other_metrics import *
import lasagne

VHRED_FOLDER = '../vhred/'
# Import VHRED files.
import sys

sys.path.insert (0, VHRED_FOLDER)
from utils import *


class ADEM (object):

    def __init__(self, config, load_from=None):

        if not load_from is None:
            self.load (load_from)
        else:
            self.config = config

        self.pretrainer = None
        if self.config['pretraining'].lower () == 'vhred':
            self.pretrainer = VHRED (self.config)

    def get_scores(self, contexts, gt_responses, model_responses):
        # Preprocess each text.

        contexts = [self.preprocessor.preprocess (s) for s in contexts]
        gt_responses = [self.preprocessor.preprocess (s) for s in gt_responses]
        model_responses = [self.preprocessor.preprocess (s) for s in model_responses]

        # Convert into format for the pretrainer.
        dataset = []
        for c, r_gt, r_m in zip (contexts, gt_responses, model_responses):
            entry = {'c': c, 'r_gt': r_gt, 'r_models': {'test': [r_m, ]}}
            dataset.append (entry)
        #  Get the embeddings through the pretrainer.
        dataset = self.pretrainer.get_embeddings (dataset, ['test', ])
        # TODO: Perform PCA.
        x = np.zeros ((len (dataset), 3, 2000), dtype=theano.config.floatX)
        for ix, entry in enumerate (dataset):
            x[ix, 0, :] = dataset[ix]['c_emb']
            x[ix, 1, :] = dataset[ix]['r_gt_emb']
            x[ix, 2, :] = dataset[ix]['r_model_embs']['test']
        x = self._apply_pca (x)
        # Get the score.
        return self._get_outputs (x)

    def _oversample(self, train_x, train_y, lengths):
        print max (lengths)
        RANGE = 9
        models = train_y.tolist ()
        bins = {}
        for y in models:
            bins[y] = {}

        # Initialize length of bins.
        for i in xrange (0, int (max (lengths)), RANGE):
            for k in bins.keys ():
                bins[k][i] = []

        # Sort examples by length.
        ix = -1
        for m, l in zip (models[:len (lengths)], lengths):
            ix += 1
            for i in xrange (int (max (lengths)) - RANGE, -1, -RANGE):
                if l > i:
                    bins[m][i].append (ix)
                    break

        # Create the new training data.
        maxes = {}
        for i in xrange (0, int (max (lengths)), RANGE):
            maxes[i] = np.max (
                [len (b) for b in [bins[1.0][i], bins[2.0][i], bins[3.0][i], bins[4.0][i], bins[5.0][i]]])
        n = int (np.sum (maxes.values ())) * 5

        new_x = np.zeros ((n, train_x.shape[1], train_x.shape[2]), dtype='float32')
        new_y = np.zeros ((n,), dtype='float32')
        new_l = np.zeros ((n,), dtype='int32')

        new_models = []
        ix = 0  # Index in new_x
        for i in xrange (0, int (max (lengths)), RANGE):
            for model in bins.keys ():
                added = 0
                jx = 0  # Index in bins[model][i]
                while added < maxes[i]:
                    if len (bins[model][i]) == 0:
                        break
                    index = bins[model][i][jx]
                    new_x[ix, :, :] = train_x[index, :, :]
                    new_y[ix] = train_y[index]
                    new_l[ix] = lengths[index]
                    ix += 1
                    jx += 1
                    if jx == len (bins[model][i]): jx = 0
                    added += 1
        return (new_x, new_y)

    def _compute_pca(self, train_x):
        # Reduce the input vectors to a lower dimensional space.
        self.pca = PCA (n_components=self.config['pca_components'])

        # Count the number of examples in each set.
        n_train = train_x.shape[0]

        # Flatten the first two dimensions. The first dimension now includes all the contexts, then responses.
        x_flat = np.zeros ((n_train * 3, train_x.shape[2]), dtype='float32')
        for i in range (3):
            x_flat[n_train * i: n_train * (i + 1), :] = train_x[:, i, :]
        pca_train = self.pca.fit_transform (x_flat)

        print 'PCA Variance'
        print self.pca.explained_variance_ratio_
        print np.sum (self.pca.explained_variance_ratio_)

        # Expand the result back to three dimensions.
        train_pca_x = np.zeros ((n_train, 3, self.config['pca_components']), dtype='float32')
        for i in range (3):
            train_pca_x[:, i, :] = pca_train[n_train * i: n_train * (i + 1), :]

        return train_pca_x

    def _apply_pca(self, x):
        pca_x = np.zeros ((x.shape[0], 3, self.config['pca_components']), dtype='float32')
        # Perform PCA, only fitting on the training set.
        pca_x[:, 0, :] = self.pca.transform (x[:, 0, :])
        pca_x[:, 1, :] = self.pca.transform (x[:, 1, :])
        pca_x[:, 2, :] = self.pca.transform (x[:, 2, :])
        return pca_x

    '''def _create_data_splits(self, data):
        n_models = len(data[0]['r_models'])
        n = len(data)
        n_train = int((1 - (self.config['val_percent'] + self.config['test_percent'])) * n) * n_models
        n_val = int((1 - self.config['test_percent']) * n) * n_models - n_train
        n_test = n * n_models - n_train - n_val
        index_val = n_train / n_models
        index_test = (n_train + n_test) / n_models
        real_train = n_train
        emb_dim = len(data[0]['c_emb'])
        # Create arrays to store the data. The middle dimension represents:
        # 0: context, 1: gt_response, 2: model_response
        train_x = np.zeros((real_train, 3, emb_dim), dtype=theano.config.floatX)
        val_x = np.zeros((n_val, 3, emb_dim), dtype=theano.config.floatX)
        test_x = np.zeros((n_test, 3, emb_dim), dtype=theano.config.floatX)
        train_y = np.zeros((real_train,), dtype=theano.config.floatX)
        val_y = np.zeros((n_val,), dtype=theano.config.floatX)
        test_y = np.zeros((n_test,), dtype=theano.config.floatX)

        train_lengths = np.zeros((real_train,), dtype=theano.config.floatX)
        order = []
        # Load in the embeddings from the dataset.
        for ix, entry in enumerate(data):
            for jx, m_name in enumerate(data[ix]['r_models'].keys()):
                kx = ix * n_models + jx

                if kx < n_train:
                    if ix == 0:
                        order.append(m_name)
                    if kx >= real_train:
                        continue
                    train_x[kx, 0, :] = data[ix]['c_emb']
                    train_x[kx, 1, :] = data[ix]['r_gt_emb']
                    train_x[kx, 2, :] = data[ix]['r_model_embs'][m_name]
                    train_y[kx] = data[ix]['r_models'][m_name][1][0]
                    train_lengths[kx] = data[ix]['r_models'][m_name][2]

                elif kx < n_train + n_val:
                    val_x[kx - n_train, 0, :] = data[ix]['c_emb']
                    val_x[kx - n_train, 1, :] = data[ix]['r_gt_emb']
                    val_x[kx - n_train, 2, :] = data[ix]['r_model_embs'][m_name]
                    val_y[kx - n_train] = data[ix]['r_models'][m_name][1][0]

                else:
                    test_x[kx - n_train - n_val, 0, :] = data[ix]['c_emb']
                    test_x[kx - n_train - n_val, 1, :] = data[ix]['r_gt_emb']
                    test_x[kx - n_train - n_val, 2, :] = data[ix]['r_model_embs'][m_name]
                    test_y[kx - n_train - n_val] = data[ix]['r_models'][m_name][1][0]

        return train_x, test_x, val_x, train_y, test_y, val_y, train_lengths, index_val, index_test, order'''

    def _create_data_splits(self, data):
        n_models = len (data[0]['r_models'])
        #n = len (data) * n_models
        #n_train = int ((1 - (self.config['val_percent'] + self.config['test_percent'])) * n)
        #n_val = int ((1 - self.config['test_percent']) * n) - n_train
        #n_test = n - n_train - n_val
        n = len (data)
        n_train = int ((1 - (self.config['val_percent'] + self.config['test_percent'])) * n) * n_models
        n_val = int ((1 - self.config['test_percent']) * n) * n_models - n_train
        n_test = n * n_models - n_train - n_val
        index_val = (n_val) / n_models
        index_test = (n_val + n_test) / n_models

        emb_dim = len (data[0]['c_emb'])
        # Create arrays to store the data. The middle dimension represents:
        # 0: context, 1: gt_response, 2: model_response
        train_x = np.zeros ((n_train, 3, emb_dim), dtype=theano.config.floatX)
        val_x = np.zeros ((n_val, 3, emb_dim), dtype=theano.config.floatX)
        test_x = np.zeros ((n_test, 3, emb_dim), dtype=theano.config.floatX)
        train_y = np.zeros ((n_train,), dtype=theano.config.floatX)
        val_y = np.zeros ((n_val,), dtype=theano.config.floatX)
        test_y = np.zeros ((n_test,), dtype=theano.config.floatX)

        train_lengths = np.zeros ((n_train,), dtype=theano.config.floatX)

        order = []
        # Load in the embeddings from the dataset.
        for ix, entry in enumerate (data):
            for jx, m_name in enumerate (data[ix]['r_models'].keys ()):
                kx = ix * n_models + jx

                if kx < n_val:
                    if ix == 0: order.append (m_name)
                    val_x[kx, 0, :] = data[ix]['c_emb']
                    val_x[kx, 1, :] = data[ix]['r_gt_emb']
                    val_x[kx, 2, :] = data[ix]['r_model_embs'][m_name]
                    val_y[kx] = data[ix]['r_models'][m_name][1][0]

                elif kx < n_val + n_test:
                    test_x[kx - n_val, 0, :] = data[ix]['c_emb']
                    test_x[kx - n_val, 1, :] = data[ix]['r_gt_emb']
                    test_x[kx - n_val, 2, :] = data[ix]['r_model_embs'][m_name]
                    test_y[kx - n_val] = data[ix]['r_models'][m_name][1][0]

                else:
                    train_x[kx - n_test - n_val, 0, :] = data[ix]['c_emb']
                    train_x[kx - n_test - n_val, 1, :] = data[ix]['r_gt_emb']
                    train_x[kx - n_test - n_val, 2, :] = data[ix]['r_model_embs'][m_name]
                    train_y[kx - n_test - n_val] = data[ix]['r_models'][m_name][1][0]
                    train_lengths[kx - n_test - n_val] = data[ix]['r_models'][m_name][2]
        return train_x, val_x, test_x, train_y, val_y, test_y, train_lengths, index_val, index_test, order

    def _create_data_splits_unsupervised(self, data):
        n_models = len (data[0]['r_models'])
        #n = len (data) * n_models
        #n_train = int ((1 - (self.config['val_percent'] + self.config['test_percent'])) * n)
        #n_val = int ((1 - self.config['test_percent']) * n) - n_train
        #n_test = n - n_train - n_val
        n = len (data)
        n_train = int ((1 - (self.config['val_percent'] + self.config['test_percent'])) * n) * n_models
        n_val = int ((1 - self.config['test_percent']) * n) * n_models - n_train
        n_test = n * n_models - n_train - n_val
        index_val = (n_val) / n_models
        index_test = (n_val + n_test) / n_models

        # Create arrays to store the data. The middle dimension represents:
        # 0: context, 1: gt_response, 2: model_response
        train_y = np.zeros ((n_train,), dtype=theano.config.floatX)
        val_y = np.zeros ((n_val,), dtype=theano.config.floatX)
        test_y = np.zeros ((n_test,), dtype=theano.config.floatX)

        order = []
        # Load in the embeddings from the dataset.
        for ix, entry in enumerate (data):
            for jx, m_name in enumerate (data[ix]['r_models'].keys ()):
                kx = ix * n_models + jx

                if kx < n_val:
                    if ix == 0: order.append (m_name)
                    val_y[kx] = data[ix]['r_models'][m_name][1][0]

                elif kx < n_val + n_test:
                    test_y[kx - n_val] = data[ix]['r_models'][m_name][1][0]

                else:
                    train_y[kx - n_test - n_val] = data[ix]['r_models'][m_name][1][0]
        return train_y, val_y, test_y, index_val, index_test, order


    def _build_model(self, emb_dim, init_mean, init_range, training_mode=False):
        ## batch index
        index = T.lscalar ()
        # Theano variables for computation graph.
        x = T.tensor3 ('x')
        y = T.ivector ('y')

        # Matrices for predicting score
        self.M = theano.shared (np.eye (emb_dim).astype (theano.config.floatX), borrow=True)
        self.N = theano.shared (np.eye (emb_dim).astype (theano.config.floatX), borrow=True)
        # self.P = theano.shared(np.eye(emb_dim*2+1).astype(theano.config.floatX), borrow=True)
        # self.new_mean = theano.shared(value=init_mean.astype(theano.config.floatX), name="mean")
        # self.new_range = theano.shared(value=init_range.astype(theano.config.floatX), name="range")
        self.rng = np.random.RandomState (1)
        # self.hidden_size = 50
        # self.W1 = theano.shared(value=NormalInit(self.rng, 3 * emb_dim, self.hidden_size), name='W1')
        # self.b1 = theano.shared(value=np.zeros((self.hidden_size,), dtype='float32'), name='b1')
        # self.W2 = theano.shared(value=NormalInit(self.rng, self.hidden_size, 1), name='W2')
        # self.b2 = theano.shared(value=0., name='b2')

        # Set embeddings by slicing tensor
        self.emb_context = x[:, 0, :]
        self.emb_true_response = x[:, 1, :]
        self.emb_response = x[:, 2, :]
        # embs = T.concatenate([self.emb_context, self.emb_true_response, self.emb_response], axis = 1)
        # t1 = T.tanh(T.dot(embs, self.W1) + self.b1)
        # out = T.nnet.sigmoid(T.dot(t1, self.W2) + self.b2)
        # out = T.flatten(out)
        # output = out * 4 + 1

        # Compute score predictions
        self.pred = 0
        self.pred1 = T.sum (self.emb_context * T.dot (self.emb_response, self.M), axis=1)
        self.pred2 = T.sum (self.emb_true_response * T.dot (self.emb_response, self.N), axis=1)
        # self.pred3 = T.sum(self.emb_context * T.dot(self.emb_true_response, self.N), axis=1)

        # self.pred = output
        if self.config['use_c']: self.pred += self.pred1
        if self.config['use_r']: self.pred += self.pred2

        # ctr = T.concatenate([self.emb_context, self.emb_true_response, self.pred3.dimshuffle(0, 'x')], axis = 1)
        # cr = T.concatenate([self.emb_context, self.emb_response, self.pred1.dimshuffle(0, 'x')], axis = 1)
        # self.pred = T.sum(ctr * T.dot(cr, self.P), axis=1)

        # To re-scale dot product values to [1,5] range.
        # new_mean = T.mean(self.pred)
        # new_range = T.max(self.pred) - T.min(self.pred)
        # output = 3 + 4 * (self.pred - self.new_mean) / self.new_range
        output = T.nnet.sigmoid (self.pred) * 4 + 1
        # output = 3 + 4 * (self.pred - init_mean) / init_range
        # output = self.pred

        loss = T.mean ((output - y) ** 2)
        l2_reg = self.M.norm (2) + self.N.norm (2)  # + self.P.norm(2)
        l1_reg = self.M.norm (1) + self.N.norm (1)  # + self.P.norm(1)
        # l2_reg = self.W1.norm(2) + self.W2.norm(2)
        # l1_reg = self.W1.norm(1) + self.W2.norm(1)

        score_cost = loss + self.config['l2_reg'] * l2_reg + self.config['l1_reg'] * l1_reg

        # Get the test predictions.
        self._get_outputs = theano.function (
            inputs=[x, ],
            outputs=[output, self.pred],
            on_unused_input='warn'
        )

        params = []
        if self.config['use_c']: params.append (self.M)
        if self.config['use_r']: params.append (self.N)
        # params.append(self.P)
        # params += [self.W1, self.W2, self.b1, self.b2]
        updates = lasagne.updates.adam (score_cost, params, learning_rate=0.0001)

        if training_mode == True:
            bs = self.config['bs']
            # d = {self.new_mean: new_mean,
            # self.new_range: new_range}
            self._train_model = theano.function (
                inputs=[index],
                outputs=score_cost,
                updates=updates,
                givens={
                    x: self.train_x[index * bs: (index + 1) * bs],
                    y: self.train_y[index * bs: (index + 1) * bs],
                },
                on_unused_input='warn'
            )


    def _compute_init_values(self, emb):
        prod_list = []
        for i in xrange (len (emb)):
            term = 0
            if self.config['use_c']: term += np.dot (emb[i, 0], emb[i, 2])
            if self.config['use_r']: term += np.dot (emb[i, 1], emb[i, 2])
            prod_list.append (term)
        alpha = np.mean (prod_list)
        beta = max (prod_list) - min (prod_list)
        return alpha, beta


    def _correlation(self, output, score):
        return [spearmanr (output, score), pearsonr (output, score)]


    def _set_shared_variable(self, x):
        return theano.shared (np.asarray (x, dtype=theano.config.floatX), borrow=True)


    def train_eval(self, data, config, use_saved_embeddings=True, model=None):
        # Each dictionary looks like { 'c': context, 'r_gt': true response, 'r_models': {'hred': (model_response, score), ... }}
        fname_embeddings = '%s/%s' % (self.config['exp_folder'], self.config['vhred_embeddings_file'])
        if (not use_saved_embeddings) or (not os.path.exists (fname_embeddings)):
            # Get embeddings for our dataset.
            assert not self.pretrainer is None
            data = self.pretrainer.change_embeddings(data, config['iclr_embs'])  # config['iclr_embs']

            with open (fname_embeddings, 'wb') as handle:
                cPickle.dump (data, handle)
        else:
            with open (fname_embeddings, 'rb') as handle:
                data = cPickle.load (handle)

        # Create train, validation, and test sets.
        train_x, val_x, test_x, train_y, val_y, test_y, train_lengths, index_val, index_test, order = self._create_data_splits (
            data)

        # Oversample training set, create dataset.
        # train_x, train_y = self._oversample(train_x, train_y, train_lengths)
        # Perform PCA.
        # train_x = self._compute_pca(train_x)
        # val_x = self._apply_pca(val_x)
        # test_x = self._apply_pca(test_x)
        init_mean, init_range = self._compute_init_values (train_x)
        self.init_mean, self.init_range = init_mean, init_range

        self.train_x = self._set_shared_variable (train_x)
        self.val_x = self._set_shared_variable (val_x)
        self.test_x = self._set_shared_variable (test_x)

        self.train_y = theano.shared (np.asarray (train_y, dtype='int32'), borrow=True)

        n_train_batches = train_x.shape[0] / self.config['bs']

        # Build the Theano model.
        self._build_model (train_x.shape[2], init_mean, init_range, training_mode=True)
        if model:
            self.M.set_value(model.M.get_value())
            self.N.set_value(model.N.get_value())

        # Train the model.
        print 'Starting training...'
        epoch = 0
        # Vairables to keep track of the best achieved so far.
        best_output_val = np.zeros ((50,))
        best_val_cor, best_test_cor = [0, 0], [0, 0]
        # Keep track of loss/epoch.
        loss_list = []
        # Keep track of best parameters so far.
        best_val_loss, best_epoch = np.inf, -1

        indices = range (n_train_batches)

        model_train_out, train_pred = self._get_outputs (train_x)
        model_val_out, val_pred = self._get_outputs (val_x)
        # Get the training and validation MSE.
        train_loss = np.sqrt (np.mean (np.square (model_train_out - train_y)))
        val_loss = np.sqrt (np.mean (np.square (model_val_out - val_y)))
        # Keep track of the correlations.
        train_correlation = self._correlation (model_train_out, train_y)
        # print self.init_mean, self.init_range
        best_val_cor = self._correlation (model_val_out, val_y)
        model_out_test, _ = self._get_outputs(test_x)
        best_test_cor = self._correlation(model_out_test, test_y)
        print 'epoch: ', epoch, train_loss, val_loss, best_val_cor[0][0], best_val_cor[1][0], best_test_cor[0][0], best_test_cor[1][0]

        while (epoch < self.config['max_epochs']):
            epoch += 1
            np.random.shuffle (indices)

            # Train for an epoch.
            cost_list = []
            for i, minibatch_index in enumerate (indices):
                minibatch_cost = self._train_model (minibatch_index)
                cost_list.append (minibatch_cost)
            loss_list.append (np.mean (cost_list))

            # Get the predictions for each dataset.
            model_train_out, train_pred = self._get_outputs (train_x)
            model_val_out, val_pred = self._get_outputs (val_x)
            # Get the training and validation MSE.
            train_loss = np.sqrt (np.mean (np.square (model_train_out - train_y)))
            val_loss = np.sqrt (np.mean (np.square (model_val_out - val_y)))
            # Keep track of the correlations.
            train_correlation = self._correlation (model_train_out, train_y)
            # print self.init_mean, self.init_range
            val_cor = self._correlation (model_val_out, val_y)
            print 'epoch: ', epoch, train_loss, val_loss, val_cor[0][0], val_cor[1][0]
            # print train_pred[:10], model_train_out[:10], val_pred[:10], model_val_out[:10]

            # Only save the model when we best the best MSE on the validation set.
            if val_cor[1][0] + val_cor[0][0] > best_val_cor[1][0] + best_val_cor[0][0]:
                best_val_cor = self._correlation (model_val_out, val_y)
                best_val_loss = val_loss
                best_output_val = model_val_out
                #print 'valid data: '


                model_out_test, _ = self._get_outputs (test_x)
                best_test_cor = self._correlation (model_out_test, test_y)
                best_test_loss = np.sqrt (np.mean (np.square (model_out_test - test_y)))
                print epoch, best_test_loss, best_test_cor[0][0], best_test_cor[1][0]
                #print 'test data: '
                #calculate_sentence_correlation(index_val, index_test, config)
                #calculate_model_correlation (index_val, index_test, config, score=model_out_test, order=order)

                best_epoch = epoch
                self.best_params = [self.M.get_value (), self.N.get_value ()]
            # self.best_params = [self.W1.get_value(), self.W2.get_value()]

        print 'Done training!'
        print 'Last updated on epoch %d' % best_epoch

        print 'bleu on valid: '
        # calculate_model_correlation(index_val / 4, index_test / 4, config)
        calculate_sentence_correlation (0, index_val, config)
        print 'bleu on test: '
        #calculate_model_correlation(index_val, index_test, config)
        calculate_sentence_correlation (index_val, index_test, config)

        # Print out results
        results = [('TRAIN', train_correlation[1], train_correlation[0], train_loss),
                   ('VAL', best_val_cor[1], best_val_cor[0], best_val_loss),
                   ('TEST', best_test_cor[1], best_test_cor[0], best_test_loss)]

        print_string = ''
        for name, p, s, rmse in results:
            print_string += '\n%s Pearson: %.3f (%.3f)\tSpearman: %.3f (%.3f)\tRMSE: %.3f' % (
                name, p[0], p[1], s[0], s[1], rmse)
        print print_string


    def load(self, f_model):
        with open (f_model, 'rb') as handle:
            saved_model = cPickle.load (handle)
        self.config = saved_model['config']
        '''for kv in self.config.items():
            print kv'''
        init_mean, init_range = saved_model['init_mean'], saved_model['init_range']
        self._build_model (self.config['pca_components'], init_mean, init_range)
        self.pca = saved_model['pca']
        self.M.set_value (saved_model['params'][0])
        self.N.set_value (saved_model['params'][1])


    def save(self):
        # Save the PCA model.
        saved_model = {#'pca': self.pca,
                       'params': self.best_params,
                       'config': self.config}
                       #'init_mean': self.init_mean,
                       #'init_range': self.init_range}
        with open ('%s/adem_model.pkl' % self.config['exp_folder'], 'wb') as handle:
            cPickle.dump (saved_model, handle)
