import os
import argparse
import modeling
import tokenization
import numpy as np
import tensorflow as tf
from run_classifier import ColaProcessor, SstProcessor, MrpcProcessor, QqpProcessor, QnliProcessor, MnliProcessor, RteProcessor, SnliProcessor, WnliProcessor
from run_classifier import file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features


def evaluate_split(input_file, estimator, max_seq_length=128):
    input_fn = file_based_input_fn_builder(
        input_file=input_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.evaluate(input_fn=input_fn, steps=None)
    for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))


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

def grade(basepath, result, gold):
    answer_list = []
    for prediction in result:
        pred = 1 if prediction[1] > prediction[0] else 0
        answer_list.append(pred)

    task = 'SST-2'
    data_dir = '%s/glue_data/%s' % (basepath, task)
    problems = '%s/test.tsv' % data_dir

    y_hat = []
    y = []
    with open(problems, 'rt') as handle:
        all_line = handle.read()
        for idx, line in enumerate(all_line.split('\n')):
            if idx==0:
                continue
            if len(line) > 0:
                y.append(gold[line.split('\t')[1]])
                y_hat.append(answer_list[idx - 1])

    y = np.array(y)
    y_hat = np.array(y_hat)

    acc = np.mean(y==y_hat)
    print('accuracy=%f' % acc)


def evaluate(basepath, bert_type, task):
    save_checkpoints_steps = 1000
    iterations_per_loop = 1000
    num_tpu_cores = 8
    learning_rate = 5e-5
    max_seq_length = 128

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_pretrained_dir = '%s/%s' % (basepath, bert_type)
    data_dir = '%s/glue_data/%s' % (basepath, task)
    output_dir = bert_pretrained_dir
    bert_config_file = '%s/bert_config.json' % (bert_pretrained_dir)
    # init_checkpoint = '%s/bert_model.ckpt' % (bert_pretrained_dir)
    init_checkpoint = '%s/bert/%s/' % (data_dir, bert_type)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=init_checkpoint,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
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

    # dev
    # dev_examples = processor.get_dev_examples(data_dir)

    # trn_file = os.path.join(data_dir, "trn.%s.tf_record" % bert_type)
    # evaluate_split(trn_file, estimator, max_seq_length)

    dev_file = os.path.join(data_dir, "dev.%s.tf_record" % bert_type)
    evaluate_split(dev_file, estimator, max_seq_length)

    tst_file = os.path.join(data_dir, "tst.%s.tf_record" % bert_type)
    predict_input_fn = file_based_input_fn_builder(
        input_file=tst_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=False)
    result = estimator.predict(input_fn=predict_input_fn)

    gold = get_gold()
    grade(basepath, result, gold)



    # embeddings = estimator.get_variable_value(estimator.get_variable_names()[4])
    # print(output_dir + '/embedding.cpkl')
    # with open(output_dir + '/embedding.cpkl', 'wb') as handle:
    #     cPickle.dump(embeddings, handle)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-t', default='SST-2', type=str,
                              choices=['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI',
                                       'QNLI', 'RTE', 'WNLI', 'SNLI', 'STS-B'])  # GLUE task type
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16']) # bert model type
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    evaluate(args.p, args.b, args.t)