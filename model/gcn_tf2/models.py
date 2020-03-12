from    layers import *
from    metrics import *
from    config import args, params

class GCN(keras.Model):

    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 feature, label, adj_list, adj_mask, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        self.feature = tf.convert_to_tensor(feature)
        self.label = tf.convert_to_tensor(label)
        self.adj_list = tf.convert_to_tensor(adj_list)
        self.adj_mask = tf.convert_to_tensor(adj_mask, dtype=tf.float32)


        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layer1 = GraphConvolution(input_dim=self.input_dim, # 1433
                                            output_dim=params['hidden1'], # 16
                                            num_features_nonzero=num_features_nonzero,
                                            activation=tf.nn.relu,
                                            dropout=args.dropout)


        self.layer2  = GraphConvolution(input_dim=params['hidden1'], # 16
                                            output_dim=self.output_dim, #params['hidden2'],
                                            num_features_nonzero=num_features_nonzero,
                                            activation=lambda x:x,
                                            dropout=args.dropout)

        self.dense_layers = []
        self.layers_ = []
        self.layers_.append(self.layer1)
        self.layers_.append(self.layer2)

        for p in self.trainable_variables:
            print(p.name, p.shape)

    def call(self, inputs, training=None):
        """
        :param inputs:
        :param training:
        :return:
        """
        x = inputs
        x_feature = tf.gather(self.feature,x)

        label = tf.gather(self.label,x)
        nbs = tf.reshape(tf.gather(self.adj_list, x),[-1])
        mask = tf.gather(self.adj_mask,x)
        degree_mask = tf.reduce_sum(mask,axis=-1)
        mask = tf.expand_dims(tf.reshape(mask,[-1]),-1)

        nbs2 = tf.reshape(tf.gather(self.adj_list, nbs),[-1])

        mask2 = tf.gather(self.adj_mask,nbs)
        degree_mask2 = tf.reduce_sum(mask2,axis=-1)
        mask2 = tf.expand_dims(tf.reshape(mask2,[-1]),-1)

        nbs_feature = tf.gather(self.feature,nbs)
        nbs2_feature = tf.gather(self.feature,nbs2)

        nbs_feature = tf.multiply(nbs_feature,mask)
        nbs2_feature= tf.multiply(nbs2_feature,mask2)

        nbs_h1 = self.layer1((nbs_feature,nbs2_feature, degree_mask2),training) #  BKH we need a mask here
        self_h1 = self.layer1((x_feature,nbs_feature,degree_mask ),training) # BH

        nbs_h1 = tf.multiply(nbs_h1,mask)
        self_h2 = self.layer2((self_h1,nbs_h1, degree_mask),training)

        outputs = [self_h2]
        for layer in self.dense_layers:
            hidden = layer(outputs[-1], training)
            outputs.append(hidden)
        output = outputs[-1]

        # # Weight decay loss
        loss = tf.zeros([])
        for layer in self.layers_:
            for var in layer.trainable_variables:
                loss += params['weight_decay'] * tf.nn.l2_loss(var)

        # Cross entropy error
        loss += softmax_cross_entropy(output, label)

        acc = accuracy(output, label)

        return loss, acc
