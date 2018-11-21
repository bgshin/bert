import os
import argparse
import modeling
import cPickle
import collections
import numpy as np
import tensorflow as tf
from run_classifier import ColaProcessor, MnliProcessor, MrpcProcessor, SstProcessor, XnliProcessor
from run_classifier import file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features
from tqdm import tqdm

def get_model_size(ckpt_fpath):
    # Open TensorFlow ckpt
    reader = tf.train.NewCheckpointReader(ckpt_fpath)

    print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
    param_map = reader.get_variable_to_shape_map()
    total_count = 0
    for k, v in param_map.items():
        if 'Momentum' not in k and 'global_step' not in k:
            temp = np.prod(v)
            total_count += temp
            print('%s: %s => %d' % (k, str(v), temp))

    print('Total Param Count: %d' % total_count)


def get_xy(model, filename):
    max_seq_length = 128
    eval_steps = None
    use_tpu = False
    eval_drop_remainder = True if use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=filename,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result_gen = model.predict(input_fn=eval_input_fn)
    result = [v for v in tqdm(result_gen)]

    with open(filename.replace('tf_record', 'cpkl'), 'rb') as handle:
        feature_list = cPickle.load(handle)

    x = np.array([item.input_ids for item in feature_list])
    y = np.array([item.label_id for item in feature_list])
    ysm = np.array(result)

    return (x, y, ysm)


def get_gold():
    truths = '../../../data/sentiment_analysis/sstb/tst'
    gold = {}
    with open(truths, 'rt') as handle:
        line = handle.read()
        for item in line.split('\n'):
            if len(item) > 0:
                label = item[0]
                string = item[2:]
                if string == 'no. .':
                    string = 'no . .'

                # if '\\/' in string:
                #     print(1)
                string = string.replace('favour', 'favor')
                string = string.replace('favourite', 'favorite')
                string = string.replace('learnt', 'learned')
                string = string.replace('humour', 'humor')
                string = string.replace('humorless and dull', 'humourless and dull')
                string = string.replace('women just wanna', 'women just wan na')
                string = string.replace('\\/', '/')
                string = string.replace('\\*', '*')
                string = string.replace('-lrb-', '(').replace('-rrb-', ')')

                gold[string] = int(label)

    return gold


def get_xy_tst(model, filename):
    max_seq_length = 128
    eval_steps = None
    use_tpu = False
    eval_drop_remainder = True if use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=filename,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result_gen = model.predict(input_fn=eval_input_fn)
    result = [v for v in tqdm(result_gen)]

    with open(filename.replace('tf_record', 'cpkl'), 'rb') as handle:
        feature_list = cPickle.load(handle)

    gold = get_gold()

    y_list = []
    for item in feature_list:
        y = gold[item.example.text_a.encode('utf-8')]
        y_list.append(y)

    x = np.array([item.input_ids for item in feature_list])
    y = np.array(y_list)
    ysm = np.array(result)

    yh = np.array([0 if a[0] > a[1] else 1 for a in ysm])
    print(np.mean(y==yh))

    return (x, y, ysm)

def generate_meterials(basepath, task, bert_type):
    bert_pretrained_dir = '%s/%s' % (basepath, bert_type)
    bert_config_file = '%s/bert_config.json' % (bert_pretrained_dir)
    data_dir = '%s/glue_data/%s' % (basepath, task)
    init_checkpoint = '%s/bert/%s/' % (data_dir, bert_type)
    learning_rate = 2e-5
    max_seq_length = 128

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "sst-2": SstProcessor,
    }
    processor = processors[task.lower()]()
    # label_list = processor.get_labels()

    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=init_checkpoint,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=10,
        num_warmup_steps=10,
        use_tpu=False,
        use_one_hot_embeddings=False)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=32,
        eval_batch_size=8,
        predict_batch_size=8)

    print('saving %s...' % estimator.get_variable_names()[12])
    embeddings = estimator.get_variable_value(estimator.get_variable_names()[12])
    filename = os.path.join(data_dir, "embedding.%s.cpkl" % bert_type)
    print(filename)
    with open(filename, 'wb') as handle:
        cPickle.dump(embeddings, handle)

    # print('======================= trn =========================')
    # filename = os.path.join(data_dir, "trn.distill.%s.tf_record" % bert_type)
    # (x_trn, y_trn, ysm_trn) = get_xy(estimator, filename)
    # with open(filename.replace('tf_record', 'materials.cpkl'), 'wb') as handle:
    #     cPickle.dump((x_trn, y_trn, ysm_trn), handle)
    #
    # print('======================= dev =========================')
    # filename = os.path.join(data_dir, "dev.distill.%s.tf_record" % bert_type)
    # (x_dev, y_dev, ysm_dev) = get_xy(estimator, filename)
    # with open(filename.replace('tf_record', 'materials.cpkl'), 'wb') as handle:
    #     cPickle.dump((x_dev, y_dev, ysm_dev), handle)
    #
    # print('======================= tst =========================')
    # filename = os.path.join(data_dir, "tst.distill.%s.tf_record" % bert_type)
    # (x_tst, y_tst, ysm_tst) = get_xy_tst(estimator, filename)
    # with open(filename.replace('tf_record', 'materials.cpkl'), 'wb') as handle:
    #     cPickle.dump((x_tst, y_tst, ysm_tst), handle)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-t', default='SST-2', type=str,
                        choices=['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI',
                                 'QNLI', 'RTE', 'WNLI', 'SNLI', 'STS-B'])  # GLUE task type
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-g', default='0')  # gpunum
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    generate_meterials(args.p, args.t, args.b)
    # init_checkpoint = '%s/%s/bert_model.ckpt' % (args.p, args.b)
    # get_model_size(init_checkpoint)