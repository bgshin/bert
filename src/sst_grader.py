import os
import argparse
import numpy as np
import tokenization
from run_classifier import ColaProcessor, SstProcessor, MrpcProcessor, QqpProcessor, QnliProcessor, MnliProcessor, RteProcessor, SnliProcessor, WnliProcessor
from run_classifier import file_based_convert_examples_to_features

def grade(basepath):
    task = 'SST-2'
    data_dir = '%s/glue_data/%s' % (basepath, task)
    problems = '%s/test.tsv' % data_dir
    answers = '%s/bert/cased_L-12_H-768_A-12/test_results.tsv' % data_dir
    truths = '../../../data/sentiment_analysis/sstb/tst'

    gold = {}
    with open(truths, 'rt') as handle:
        line = handle.read()
        for item in line.split('\n'):
            if len(item)>0:
                label = item[0]
                string = item[2:]
                if string=='no. .':
                    string='no . .'

                # if '\\/' in string:
                #     print(1)
                string = string.replace('\\/', '/')
                string = string.replace('-lrb-','(').replace('-rrb-',')')

                gold[string[:10]] = int(label)

    answer_list = []
    with open(answers, 'rt') as handle:
        all_line = handle.read()
        for line in all_line.split('\n'):
            if len(line) > 0:
                probs = [float(item) for item in line.split('\t')]
                pred = 1 if probs[1]>probs[0] else 0
                answer_list.append(pred)

    y_hat = []
    y = []
    with open(problems, 'rt') as handle:
        all_line = handle.read()
        for idx, line in enumerate(all_line.split('\n')):
            if idx==0:
                continue
            if len(line) > 0:
                y.append(gold[line.split('\t')[1][:10]])
                y_hat.append(answer_list[idx - 1])

    y = np.array(y)
    y_hat = np.array(y_hat)
    print(1)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    args = parser.parse_args()

    grade(args.p)
