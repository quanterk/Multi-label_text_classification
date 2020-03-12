from    inits import *
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers
from    config import args, params




# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res




class Dense(layers.Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, num_features_nonzero = None, bias=False, activation=tf.nn.relu,featureless = False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.type = 'dense'
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.dropout = dropout


        # helper variable for sparse dropout
        self.num_features_nonzero = num_features_nonzero
        self.weights_ = []
        w = self.add_weight('weight', [input_dim, output_dim])
        self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])


    def call(self, inputs, training= None):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, self.dropout)

        # transform
        output = dot(x, self.weights_[0], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.bias

        return self.act(output)


class GraphConvolution(layers.Layer):
    """
    Graph convolution layer.
    """
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.type = 'gcn'

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero
        self.output_dim = output_dim
        self.weights_ = []
        for i in range(1):
            w = self.add_weight('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])


    def call(self, inputs, training=None):

        # X is a batch of nodes. B*F
        # support is a batch nbs B * K *F
        x, support_, degree_mask = inputs

        nbs_size = int(support_.shape[0]/x.shape[0])
        # dropout
        if training is not False:
            x = tf.nn.dropout(x, self.dropout)


        # convolve
        if not self.featureless: # if it has features x
            pre_sup = dot(x, self.weights_[0], sparse=self.is_sparse_inputs)
        else:
            pre_sup = self.weights_[0]

        nbs_trans = dot(support_, self.weights_[0], sparse=self.is_sparse_inputs)
        self_vecs = dot(x, self.weights_[0], sparse=self.is_sparse_inputs)

        neigh_vecs = tf.reshape(nbs_trans,shape=(x.shape[0],nbs_size,self.output_dim))
        summ = tf.reduce_sum(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
        degree_mask += 1.0
        degree_mask = tf.stop_gradient(degree_mask)
        degree_mask = tf.expand_dims(degree_mask,-1)
        means = summ/(degree_mask) #self edges maybe a bug here?
        output = means
        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)
