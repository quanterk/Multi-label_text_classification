
import  tensorflow as tf
from    tensorflow.keras import optimizers
import numpy as np

seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)

from    utils import *
from    models import GCN
from    config import args, params
from data_loader import data_loader
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('tf version:', tf.__version__)
assert tf.__version__.startswith('2.')
import time
check_file = './checkpoint1'

# import logging
# logging.getLogger('tensorflow').disabled = True


def main():

    adj, features, all_labels, train, val, test = load_data(args.dataset, task_type=args.task_type)
    whole_batch = args.whole_batch
    print('adj:', adj.shape)
    print('features:', features.shape)

    MAX_NBS = params['max_degree']
    # padding adj to N*K, where K is the number of nbs
    adj_list = get_adj_list(adj, MAX_NBS)

    adj_mask = adj_list + 1
    adj_mask[adj_mask > 0] = 1

    features = preprocess_features(features)  # [49216, 2], [49216], [2708, 1433]

    # add a row
    fea = np.asarray(features)
    fea = np.insert(fea, -1, 0, axis=0)
    fea.reshape((features.shape[0] + 1, features.shape[-1]))
    features = fea

    dl = data_loader(features, adj_list, train, val, test)

    tf.keras.backend.clear_session()
    model = GCN(input_dim=features.shape[1], output_dim=all_labels.shape[1], num_features_nonzero=features[1].shape, \
                feature=features, label=all_labels, adj_list=adj_list, adj_mask=adj_mask)  # [1433]

    optimizer = optimizers.Adam(lr=params['learning_rate'])

    persist = 0
    best_val_acc = 0

    for epoch in range(args.epochs):

        if whole_batch:
            with tf.GradientTape() as tape:
                train_loss, train_acc = model(train, training=True)
            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        else:
            while not dl.train_end():
                batch = dl.get_train_batch(batch_size=params['train_batch_size'])
                with tf.GradientTape() as tape:
                    train_loss, train_acc = model(batch, training=True)

                grads = tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_loss = 0
        val_acc = 0
        if whole_batch:
            val_loss, val_acc = model(val)

        else:
            while not dl.val_end():
                batch = dl.get_val_batch(batch_size=args.val_batch_size)
                loss, acc = model(batch, training=False)
                val_loss += loss*len(batch)
                val_acc += acc*len(batch)
            val_acc/=len(val)
            val_loss/=len(val)

        if val_acc>best_val_acc:
            best_val_acc = val_acc
            persist = 0
            model.save_weights(check_file)
        else:
            persist+=1

        if persist>args.early_stopping:
            break

        if epoch % 10 == 0:
            print(epoch, float(train_loss), float(train_acc), '\tval:', float(val_acc))

    print('train done')
    model.load_weights(check_file)
    print('read checkpoint done')
    test_loss = 0
    test_acc = 0
    if whole_batch:
        test_loss, test_acc = model(test)

    else:
        while not dl.test_end():
            batch = dl.get_test_batch(batch_size=args.test_batch_size)
            loss, acc = model(batch, training=False)
            test_loss += loss * len(batch)
            test_acc += acc * len(batch)
        test_acc /= len(test)
        test_loss /= len(test)
    print('final results',test_acc.numpy())

if __name__ == '__main__':
    main()
