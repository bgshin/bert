import os
import argparse
import tokenization
from run_classifier import ColaProcessor, SstProcessor, MrpcProcessor, QqpProcessor, QnliProcessor, MnliProcessor, RteProcessor, SnliProcessor, WnliProcessor
from run_classifier import file_based_convert_examples_to_features


def save_embedding(basepath, task, bert_type):
    max_seq_length = 128

    data_dir = '%s/glue_data/%s' % (basepath, task)
    vocab_file = '%s/%s/vocab.txt' % (basepath, bert_type)
    do_lower_case = True if bert_type.split('_')[0] == 'uncased' else False

    processors = {
        "cola": ColaProcessor,
        "sst-2": SstProcessor,
        "mrpc": MrpcProcessor,
        # "sts-b"
        "qqp": QqpProcessor,
        "mnli": MnliProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
        # "ax"
    }

    processor = processors[task.lower()]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_examples = processor.get_train_examples(data_dir)
    trn_file = os.path.join(data_dir, "trn.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, trn_file)

    eval_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(data_dir, "dev.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, eval_file)

    tst_examples = processor.get_test_examples(data_dir)
    tst_file = os.path.join(data_dir, "tst.%s.tf_record" % bert_type)
    file_based_convert_examples_to_features(tst_examples, label_list,
                                            max_seq_length, tokenizer,
                                            tst_file)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    # task_list = ['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'SNLI', 'STS-B']
    # task_list = ['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'SNLI']
    # task_list = ['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI']
    # task_list = ['RTE', 'WNLI', 'SNLI']
    task_list = ['SST-2']
    for task in task_list:
        print('processing %s' % task)
        save_embedding(args.p, task, args.b)
