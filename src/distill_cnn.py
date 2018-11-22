import os
import tensorflow as tf
import argparse
import tokenization
from run_classifier import SstProcessor, file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features
import tensorflow as tf
import cPickle
import numpy as np
# from tensorflow.python import keras as keras
# from tensorflow.python.keras.layers import Conv1D
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
# from tensorflow.python.keras.layers.merge import Concatenate
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.callbacks import ModelCheckpoint

import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import os

from keras.layers import Conv1D, Lambda, Conv2D, MaxPooling2D, Reshape
from keras.layers import Dense, Flatten, Input, MaxPooling1D, Embedding, Dropout, TimeDistributed, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers.merge import Concatenate, Multiply, Add


class MyCallback(ModelCheckpoint):
    def __init__(self, data):
        print 'my callback init'
        super(MyCallback, self).__init__('.')

        self.x_trn, self.y_trn, _, self.x_dev, self.y_dev, _, self.x_tst, self.y_tst, _ = data
        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0

    def evaluate(self):
        yh_trn = self.model.predict(self.x_trn)
        yh_trn = np.argmax(yh_trn, axis=1)

        yh_dev = self.model.predict(self.x_dev)
        yh_dev = np.argmax(yh_dev, axis=1)

        yh_tst = self.model.predict(self.x_tst)
        yh_tst = np.argmax(yh_tst, axis=1)

        score_trn = np.mean(self.y_trn == yh_trn)
        score_dev = np.mean(self.y_dev == yh_dev)
        score_tst = np.mean(self.y_tst == yh_tst)



        if self.score_dev < score_dev:
            self.score_trn = score_trn
            self.score_dev = score_dev
            self.score_tst = score_tst

            print("\nupdated!!")

        print '\n[This Epoch]'
        print '\t'.join(map(str, [score_trn, score_dev, score_tst]))
        print '[Current Best]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))


    def on_train_end(self, logs=None):
        print '[Best:on_train_end]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

    def on_epoch_end(self, epoch, logs=None):
        self.evaluate()



def get_model(embeddings, max_seq_length=128):
    input_shape = (max_seq_length,)
    model_input = Input(shape=input_shape)
    print(model_input)
    z = Embedding(embeddings.shape[0],
                  embeddings.shape[1],
                  weights=[embeddings],
                  input_length=max_seq_length,
                  trainable=False)(model_input)
    filter_sizes = (2, 3, 4, 5)
    num_filters = 64
    dropout_prob = 0.2
    hidden_dims = 50
    batch_size = 64
    epochs = 200

    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(num_filters,
                      sz,
                      padding="valid",
                      activation="relu",
                      strides=1)(z)
        print(conv)
        # conv = MaxPooling1D(pool_size=2)(conv)
        conv = MaxPooling1D(pool_size=max_seq_length - sz + 1)(conv)
        print(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob)(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(model_input, model_output)
    # model.compile(loss=custom_loss_Hinton(y_gold=y_gold, tau=4, alpha=0.3, num_class=2),
    #                       optimizer="adam", metrics=["mse"])
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.summary()

    return model


def get_model2(embeddings, max_seq_length=128):
    input_shape = (max_seq_length,)
    model_input = Input(shape=input_shape)
    print(model_input)
    z = Embedding(embeddings.shape[0],
                  embeddings.shape[1],
                  weights=[embeddings],
                  input_length=max_seq_length,
                  trainable=False)(model_input)
    filter_sizes = (2, 3, 4, 5)
    num_filters = 64
    dropout_prob = 0.2
    hidden_dims = 50
    batch_size = 64
    epochs = 200

    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(num_filters,
                      sz,
                      padding="valid",
                      activation="relu",
                      strides=1)(z)
        print(conv)
        # conv = MaxPooling1D(pool_size=2)(conv)
        conv = MaxPooling1D(pool_size=max_seq_length - sz + 1)(conv)
        print(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob)(z)
    model_output = Dense(2, activation="softmax")(z)
    model = Model(model_input, model_output)
    # model.compile(loss=custom_loss_Hinton(y_gold=y_gold, tau=4, alpha=0.3, num_class=2),
    #                       optimizer="adam", metrics=["mse"])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    model.summary()

    return model

def train_model(data, embeddings, type):
    x_trn, y_trn, ysm_trn, x_dev, y_dev, ysm_dev, x_tst, y_tst, ysm_tst = data
    if type==0:
        model = get_model(embeddings)
    else:
        model = get_model2(embeddings)

    checkpoint = MyCallback(data)
    callbacks_list = [checkpoint]
    if type == 0:
        model.fit(x_trn, ysm_trn,
                  batch_size=2000,
                  callbacks=callbacks_list,
                  epochs=50)

    else:
        model.fit(x_trn, y_trn,
                  batch_size=2000,
                  callbacks=callbacks_list,
                  epochs=50)


def get_data(basepath, task, bert_type):
    data_dir = '%s/glue_data/%s' % (basepath, task)

    filename = os.path.join(data_dir, "trn.distill.%s.materials.cpkl" % bert_type)
    print('trn %s' % filename)
    with open(filename, 'rb') as handle:
        (x_trn, y_trn, ysm_trn) = cPickle.load(handle)

    filename = os.path.join(data_dir, "dev.distill.%s.materials.cpkl" % bert_type)
    print('dev %s' % filename)
    with open(filename, 'rb') as handle:
        (x_dev, y_dev, ysm_dev) = cPickle.load(handle)

    filename = os.path.join(data_dir, "tst.distill.%s.materials.cpkl" % bert_type)
    print('tst %s' % filename)
    with open(filename, 'rb') as handle:
        (x_tst, y_tst, ysm_tst) = cPickle.load(handle)

    filename = os.path.join(data_dir, "embedding.%s.cpkl" % bert_type)
    print('embeddings %s' % filename)
    with open(filename, 'rb') as handle:
        embeddings = cPickle.load(handle)

    data = x_trn, y_trn, ysm_trn, x_dev, y_dev, ysm_dev, x_tst, y_tst, ysm_tst

    return data, embeddings


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-m', default=0, type=int)  # base path
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g


    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    ktf.set_session(get_session())

    print('loading...')
    data, embeddings = get_data(args.p, 'SST-2', args.b)
    print('training...')
    train_model(data, embeddings, args.m)

# nohup python distill_cnn.py -m 0 -b uncased_L-24_H-1024_A-16 > m0.large.log &
# nohup python distill_cnn.py -m 1 -b uncased_L-24_H-1024_A-16 > m1.large.log &
# nohup python distill_cnn.py -m 0 -b cased_L-12_H-768_A-12 > m0.base.log &
# nohup python distill_cnn.py -m 1 -b cased_L-12_H-768_A-12 > m1.base.log &
