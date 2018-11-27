import argparse
import codecs
import time
import numpy as np
from src.keras_bert import load_trained_model_from_checkpoint
import collections
import six
import tensorflow as tf
import cPickle
import csv
import os


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
        # filename = '%s/trn.features.%s.cpkl' % (data_dir, bert_type)
        # with open(filename, 'rb') as handle:
        #     (xid_trn, xseg_trn, xmask_trn, y_trn, tokens_trn) = cPickle.load(handle)

        filename = '%s/dev.features.%s.cpkl' % (data_dir, bert_type)
        with open(filename, 'rb') as handle:
            (xid_dev, xseg_dev, xmask_dev, y_dev, tokens_dev) = cPickle.load(handle)

        filename = '%s/tst.features.%s.cpkl' % (data_dir, bert_type)
        with open(filename, 'rb') as handle:
            (xid_tst, xseg_tst, xmask_tst, y_gold, tokens_tst) = cPickle.load(handle)

    with Timer('loading from checkpt...'):
        model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
    # model.summary(line_length=120)
    model.summary()

    # model.fit(x_trn, y_trn)
    #
    # model.fit([xid_dev, xseg_dev, xmask_dev], y_dev,
    #           batch_size=1000,
    #           epochs=50)


    # model.fit([xid_trn, xseg_trn, xmask_trn], y_trn,
    #           batch_size=1000,
    #           epochs=50,
    #           validation_data=([xid_dev, xseg_dev, xmask_dev], y_dev))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-b', default='cased_L-12_H-768_A-12',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    run(args.p, args.b, 'SST-2')