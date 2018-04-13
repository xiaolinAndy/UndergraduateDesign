import theano
import numpy as np
from theano import tensor as T
from theano import config
from lasagne_average_layer import lasagne_average_layer
import lasagne

class ppdb_word_model(object):

    def __init__(self, We_initial, params):

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix()
        p1mask = T.matrix(); p2mask = T.matrix()

        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_average = lasagne_average_layer([l_emb, l_mask])

        embg1 = lasagne.layers.get_output(l_average, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_average, {l_in:g2batchindices, l_mask:g2mask})
        embp1 = lasagne.layers.get_output(l_average, {l_in:p1batchindices, l_mask:p1mask})
        embp2 = lasagne.layers.get_output(l_average, {l_in:p2batchindices, l_mask:p2mask})

        #objective function
        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2

        self.all_params = lasagne.layers.get_all_params(l_average, trainable=True)

        word_reg = 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
        cost = T.mean(cost) + word_reg

        #feedforward
        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask], cost)

        prediction = g1g2

        self.scoring_function = theano.function([g1batchindices, g2batchindices,
                             g1mask, g2mask],prediction)

        #updates
        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices,
                             g1mask, g2mask, p1mask, p2mask], cost, updates=updates)

class unsupervised_adem_model(object):

    def __init__(self, We_initial, params):

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        self.We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        self.M = theano.shared (np.eye (self.We.get_value().shape[1]).astype (theano.config.floatX), borrow=True)
        self.N = theano.shared (np.eye (self.We.get_value().shape[1]).astype (theano.config.floatX), borrow=True)

        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix(); g3batchindices = T.imatrix()
        p1batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix(); g3mask = T.matrix()
        p1mask = T.matrix()

        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.We.get_value().shape[0], output_size=self.We.get_value().shape[1], W=self.We)
        l_average = lasagne_average_layer([l_emb, l_mask])

        embg1 = lasagne.layers.get_output(l_average, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_average, {l_in:g2batchindices, l_mask:g2mask})
        embg3 = lasagne.layers.get_output(l_average, {l_in:g3batchindices, l_mask:g3mask})
        embp1 = lasagne.layers.get_output(l_average, {l_in:p1batchindices, l_mask:p1mask})

        #objective function
        crt = T.nnet.sigmoid(T.sum(embg2 * T.dot(embg3, self.M), axis=1)) #+ T.sum(embg2 * T.dot(embg3, self.N), axis=1))
        crf = T.nnet.sigmoid(T.sum(embg2 * T.dot(embp1, self.M), axis=1)) #+ T.sum(embg2 * T.dot(embp1, self.N), axis=1))

        cost = params.margin - crt + crf
        cost = cost*(cost > 0)

        #self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + [self.M, self.N]
        self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + [self.M]

        #word_reg = 0.5*params.LW*lasagne.regularization.l2(self.We-initial_We) + 0.5*params.LC*self.M.norm(2) + self.N.norm(2)
        word_reg = 0.5 * params.LW * lasagne.regularization.l2(self.We - initial_We) + 0.5 * params.LC * self.M.norm(2)
        cost = T.mean(cost) + word_reg

        #feedforward
        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask], cost, on_unused_input='warn')

        prediction = crt * 4 + 1

        self.scoring_function = theano.function([g1batchindices, g2batchindices, g3batchindices,
                             g1mask, g2mask, g3mask],prediction, on_unused_input='warn')

        #updates
        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask], cost, updates=updates, on_unused_input='warn')

class unsupervised_adem_model_complex(object):

    def __init__(self, We_initial, params):

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        self.We = theano.shared(np.asarray(We_initial, dtype = config.floatX))

        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix(); g3batchindices = T.imatrix()
        p1batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix(); g3mask = T.matrix()
        p1mask = T.matrix()

        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.We.get_value().shape[0], output_size=self.We.get_value().shape[1], W=self.We)
        l_average = lasagne_average_layer([l_emb, l_mask])

        embg1 = lasagne.layers.get_output(l_average, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_average, {l_in:g2batchindices, l_mask:g2mask})
        embg3 = lasagne.layers.get_output(l_average, {l_in:g3batchindices, l_mask:g3mask})
        embp1 = lasagne.layers.get_output(l_average, {l_in:p1batchindices, l_mask:p1mask})

        #objective function
        crt = T.concatenate([embg1, embg2, embg3], axis=1)
        crf = T.concatenate([embg1, embg2, embp1], axis=1)

        l_in2 = lasagne.layers.InputLayer((None, self.We.get_value().shape[1] * 3))
        d1 = lasagne.layers.DenseLayer(l_in2, self.We.get_value().shape[1], nonlinearity=lasagne.nonlinearities.tanh)
        l_sigmoid = lasagne.layers.DenseLayer(d1, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
        st = lasagne.layers.get_output(l_sigmoid, {l_in2: crt})
        sf = lasagne.layers.get_output(l_sigmoid, {l_in2: crf})


        cost = params.margin - st + sf
        cost = cost*(cost > 0)

        #self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + [self.M, self.N]
        self.network_params = lasagne.layers.get_all_params(l_average, trainable=True) + lasagne.layers.get_all_params(
            l_sigmoid, trainable=True)
        self.network_params.pop(0)
        self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + lasagne.layers.get_all_params(l_sigmoid, trainable=True)
        l2 = 0.5 * params.LC * sum(lasagne.regularization.l2(x) for x in self.network_params)
        #word_reg = 0.5*params.LW*lasagne.regularization.l2(self.We-initial_We) + 0.5*params.LC*self.M.norm(2) + self.N.norm(2)
        word_reg = 0.5 * params.LW * lasagne.regularization.l2(self.We - initial_We)
        cost = T.mean(cost) + word_reg + l2

        #feedforward
        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask], cost, on_unused_input='warn')

        prediction = st * 4 + 1

        self.scoring_function = theano.function([g1batchindices, g2batchindices, g3batchindices,
                             g1mask, g2mask, g3mask],prediction, on_unused_input='warn')

        #updates
        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask], cost, updates=updates, on_unused_input='warn')

class unsupervised_ruber_model(object):

    def __init__(self, We_initial, params):

        #params
        initial_We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        self.We = theano.shared(np.asarray(We_initial, dtype = config.floatX))
        self.M = theano.shared (np.eye (self.We.get_value().shape[1]).astype (theano.config.floatX), borrow=True)
        #self.N = theano.shared (np.eye (self.We.get_value().shape[1]).astype (theano.config.floatX), borrow=True)

        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix(); g3batchindices = T.imatrix()
        p1batchindices = T.imatrix()
        g1mask = T.matrix(); g2mask = T.matrix(); g3mask = T.matrix()
        p1mask = T.matrix()
        max_gg = T.scalar(); min_gg = T.scalar()

        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=self.We.get_value().shape[0], output_size=self.We.get_value().shape[1], W=self.We)
        l_average = lasagne_average_layer([l_emb, l_mask])

        rt = lasagne.layers.get_output(l_emb, {l_in: g2batchindices, l_mask: g2mask})
        rm = lasagne.layers.get_output(l_emb, {l_in: g3batchindices, l_mask: g3mask})
        pt = T.max(rt, axis=1)
        pm = T.max(rm, axis=1)
        nt = T.min(rt, axis=1)
        nm = T.min(rm, axis=1)
        wt = T.concatenate([pt, nt], axis = 1)
        wm = T.concatenate([pm, nm], axis = 1)
        g1g2 = (wt * wm).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(wt ** 2, axis=1)) * T.sqrt(T.sum(wm ** 2, axis=1))
        g1g2 = g1g2 / g1g2norm
        max_g = (max_gg >= T.max(g1g2)) * max_gg + (max_gg < T.max(g1g2)) * T.max(g1g2)
        min_g = (min_gg <= T.min(g1g2)) * min_gg + (min_gg > T.min(g1g2)) * T.min(g1g2)
        g1g2 = (g1g2 - min_g) / (max_g - min_g)



        embg1 = lasagne.layers.get_output(l_average, {l_in:g1batchindices, l_mask:g1mask})
        embg2 = lasagne.layers.get_output(l_average, {l_in:g2batchindices, l_mask:g2mask})
        embg3 = lasagne.layers.get_output(l_average, {l_in:g3batchindices, l_mask:g3mask})
        embp1 = lasagne.layers.get_output(l_average, {l_in:p1batchindices, l_mask:p1mask})

        #objective function
        crt = T.nnet.sigmoid(T.sum(embg1 * T.dot(embg3, self.M), axis=1)) #+ T.sum(embg2 * T.dot(embg3, self.N), axis=1))
        crf = T.nnet.sigmoid(T.sum(embg1 * T.dot(embp1, self.M), axis=1)) #+ T.sum(embg2 * T.dot(embp1, self.N), axis=1))

        cost = params.margin - crt + crf
        cost = cost*(cost > 0)

        #self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + [self.M, self.N]
        self.all_params = lasagne.layers.get_all_params(l_average, trainable=True) + [self.M]

        #word_reg = 0.5*params.LW*lasagne.regularization.l2(self.We-initial_We) + 0.5*params.LC*self.M.norm(2) + self.N.norm(2)
        word_reg = 0.5 * params.LW * lasagne.regularization.l2(self.We - initial_We) + 0.5 * params.LC * self.M.norm(2)
        cost = T.mean(cost) + word_reg

        #feedforward
        self.feedforward_function = theano.function([g1batchindices,g1mask], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask], cost, on_unused_input='warn')

        prediction = (crt + g1g2) * 2 + 1

        self.scoring_function = theano.function([g1batchindices, g2batchindices, g3batchindices,
                             g1mask, g2mask, g3mask, max_gg, min_gg],prediction, on_unused_input='warn')

        #updates
        grads = theano.gradient.grad(cost, self.all_params)
        if params.clip:
            grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
        updates = params.learner(grads, self.all_params, params.eta)
        self.train_function = theano.function([g1batchindices, g2batchindices, g3batchindices, p1batchindices,
                             g1mask, g2mask, g3mask, p1mask, max_gg, min_gg], [cost, max_g, min_g], updates=updates, on_unused_input='warn')