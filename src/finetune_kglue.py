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


class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data):
        super(MyCallback, self).__init__(filepath, save_weights_only=True)
        self.x_dev, self.y_dev, self.x_tst, self.y_tst = data

        # self.real_save = real_save
        self.filepath_template = self.filepath+'-%s'
        self.acc_tst = 0
        self.best_epoch = 0
        self.acc_dev = 0
        self.filepath = filepath

    def print_status(self):
        print('\n======================= [Best tst_acc (%f) (epoch = %d)] dev_acc =%f =============='
              % (self.acc_tst, self.best_epoch, self.acc_dev))


    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        self.print_status()

    def on_epoch_end(self, epoch, logs=None):
        acc_dev = self.model.evaluate(self.x_dev, self.y_dev)
        acc_tst = self.model.evaluate(self.x_tst, self.y_tst)

        print('\n======================= [current tst_acc (%f)] dev_acc =%f =============='
              % (acc_tst, acc_dev))

        if self.acc_dev < acc_dev:
            self.acc_dev = acc_dev
            self.acc_tst = acc_tst
            self.best_epoch = 1
            self.model.save_weights(self.filepath, overwrite=True)

            print('updated!!!')
            self.print_status()


def run(basepath, bert_type, task, seq_len):
    data_dir = '%s/glue_data/%s' % (basepath, task)
    bert_dir = '%s/%s' % (basepath, bert_type)
    config_path = '%s/bert_config.json' % bert_dir
    checkpoint_path = '%s/bert_model.ckpt' % bert_dir

    with Timer('laod dataset...'):
        filename = '%s/trn.features.s%d.%s.cpkl' % (data_dir, seq_len, bert_type)
        with open(filename, 'rb') as handle:
            (xid_trn, xseg_trn, xmask_trn, y_trn, tokens_trn) = cPickle.load(handle)

        filename = '%s/dev.features.s%d.%s.cpkl' % (data_dir, seq_len, bert_type)
        with open(filename, 'rb') as handle:
            (xid_dev, xseg_dev, xmask_dev, y_dev, tokens_dev) = cPickle.load(handle)

        filename = '%s/tst.features.s%d.%s.cpkl' % (data_dir, seq_len, bert_type)
        with open(filename, 'rb') as handle:
            (xid_tst, xseg_tst, xmask_tst, y_gold, tokens_tst) = cPickle.load(handle)

    # n_class = max(y_trn)+1
    n_class = max(y_dev)+1

    with Timer('loading from checkpt...'):
        model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True, n_class=n_class, n_seq=seq_len)
    model.summary(line_length=120)

    y_trn = keras.utils.to_categorical(y_trn, n_class)
    y_dev = keras.utils.to_categorical(y_dev, n_class)
    y_tst = keras.utils.to_categorical(y_gold, n_class)



    filename = "%s/finetune.%s.kmodel" % (data_dir, bert_type)
    # xid_dev = xid_dev[:12]
    # xseg_dev = xseg_dev[:12]
    # xmask_dev = xmask_dev[:12]
    # y_dev = y_dev[:12]

    # data = [xid_dev, xseg_dev, xmask_dev], y_dev, [xid_dev, xseg_dev, xmask_dev], y_dev
    data = [xid_dev, xseg_dev, xmask_dev], y_dev, [xid_tst, xseg_tst, xmask_tst], y_tst
    callbacks_list = [MyCallback(filename, data)]


    # model.fit([xid_dev, xseg_dev, xmask_dev], y_dev,
    #           batch_size=6,
    #           epochs=2,
    #           callbacks=callbacks_list
    #           )

    model.fit([xid_trn, xseg_trn, xmask_trn], y_trn,
              batch_size=6,
              epochs=50,
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
    parser.add_argument('-s', default=128, type=int)  # seqlen
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g


    def get_session(gpu_fraction=1):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                    allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


    ktf.set_session(get_session())


    run(args.p, args.b, args.t, args.s)
