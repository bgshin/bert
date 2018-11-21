import os
import argparse
import tokenization
import numpy as np
import cPickle
import collections
import tensorflow as tf
from run_classifier import SstProcessor
from run_classifier import convert_single_example

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    feature_list = []
    # x_list = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        feature_list.append(feature)
        # x_list.append(example)

    with open(output_file.replace('tf_record', 'cpkl'), 'wb') as handle:
        # cPickle.dump((x_list, feature_list), handle)
        cPickle.dump(feature_list, handle)


def to_tfrecord(basepath, task, bert_type):
    max_seq_length = 128

    data_dir = '%s/glue_data/%s' % (basepath, task)
    vocab_file = '%s/%s/vocab.txt' % (basepath, bert_type)
    do_lower_case = True if bert_type.split('_')[0] == 'uncased' else False

    processor = SstProcessor()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_examples = processor.get_train_examples(data_dir)
    trn_file = os.path.join(data_dir, "trn.distill.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, trn_file)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(data_dir, "dev.distill.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, eval_file)

    tst_examples = processor.get_test_examples(data_dir)
    tst_file = os.path.join(data_dir, "tst.distill.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(
        tst_examples, label_list, max_seq_length, tokenizer, tst_file)


def decode(record):
    seq_length = 128
    name_to_features = collections.OrderedDict()
    name_to_features["input_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
    name_to_features["input_mask"] = tf.FixedLenFeature([seq_length], tf.int64)
    name_to_features["segment_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.int64)

    example = tf.parse_single_example(record, name_to_features)

    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def to_numpy(basepath, task):
    with tf.Graph().as_default():
        data_dir = '%s/glue_data/%s' % (basepath, task)
        filepath = os.path.join(data_dir, "dev.test.tf_record")
        dataset = tf.data.TFRecordDataset(filepath)
        dataset = dataset.map(decode)

        # dataset = tf.data.TFRecordDataset(filepath).map(decode)
        dataset = dataset.batch(100)

        iterator = dataset.make_one_shot_iterator()
        lids, mask, ids, sids = iterator.get_next()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        # Create a session and evaluate `whole_dataset_tensors` to get arrays.
        with tf.Session() as sess:
            sess.run(init_op)
            whole_dataset_arrays = sess.run([lids, mask, ids, sids])

def reader(basepath, task):
    data_dir = '%s/glue_data/%s' % (basepath, task)
    trn_file = os.path.join(data_dir, "trn.test.cpkl")
    dev_file = os.path.join(data_dir, "dev.test.cpkl")
    tst_file = os.path.join(data_dir, "tst.test.cpkl")

    with open(tst_file, 'rb') as handle:
        # x_list, feature_list = cPickle.load(handle)
        feature_list = cPickle.load(handle)

    print(1)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    to_tfrecord(args.p, 'SST-2', args.b)

    # to_numpy(args.p, 'SST-2')
    # reader(args.p, 'SST-2')
