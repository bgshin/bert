import argparse
import time
from src.keras_bert import load_trained_model_from_checkpoint
import cPickle
import keras
import os
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def run(basepath, bert_type, task):
    data_dir = '%s/glue_data/%s' % (basepath, task)
    bert_dir = '%s/%s' % (basepath, bert_type)
    config_path = '%s/bert_config.json' % bert_dir
    checkpoint_path = '%s/bert_model.ckpt' % bert_dir
    vocab_path =  '%s/vocab.txt' % bert_dir
    do_lower_case = True if bert_type.split('_')[0] == 'uncased' else False

    with Timer('laod dataset...'):
        filename = '%s/trn.features.%s.cpkl' % (data_dir, bert_type)
        with open(filename, 'rb') as handle:
            (xid_trn, xseg_trn, xmask_trn, y_trn, tokens_trn) = cPickle.load(handle)

        filename = '%s/dev.features.%s.cpkl' % (data_dir, bert_type)
        with open(filename, 'rb') as handle:
            (xid_dev, xseg_dev, xmask_dev, y_dev, tokens_dev) = cPickle.load(handle)

        filename = '%s/tst.features.%s.cpkl' % (data_dir, bert_type)
        with open(filename, 'rb') as handle:
            (xid_tst, xseg_tst, xmask_tst, y_gold, tokens_tst) = cPickle.load(handle)

    # n_class = max(y_trn)+1
    n_class = max(y_dev)+1

    with Timer('loading from checkpt...'):
        model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, n_class=n_class)
    model.summary(line_length=120)

    y_trn = keras.utils.to_categorical(y_trn, n_class)
    y_dev = keras.utils.to_categorical(y_dev, n_class)
    y_tst = keras.utils.to_categorical(y_gold, n_class)

    # model.fit([xid_dev, xseg_dev, xmask_dev], y_dev,
    #           batch_size=6,
    #           epochs=50)

    filename = "%s/k-%s-{epoch:02d}-{val_acc:.2f}.hdf5" % (data_dir, bert_type)
    checkpoint = ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit([xid_trn, xseg_trn, xmask_trn], y_trn,
              batch_size=6,
              epochs=50,
              validation_data=([xid_dev, xseg_dev, xmask_dev], y_dev),
              callbacks=callbacks_list
              )




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-t', default='SST-2', type=str,
                              choices=['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI',
                                       'QNLI', 'RTE', 'WNLI', 'SNLI', 'STS-B'])  # GLUE task type
    parser.add_argument('-b', default='cased_L-12_H-768_A-12',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16']) # bert model type
    parser.add_argument('-g', default='0')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g


    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    ktf.set_session(get_session())


    run(args.p, args.b, args.t)
