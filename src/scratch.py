import os
import numpy as np
import tensorflow as tf
import collections
import tokenization
from run_classifier import SstProcessor, file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features

import modeling
import cPickle

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
# 324,935,430 ~324M

# tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
# result = tokenizer.tokenize(u" \tHeLLo!how  \n Are yoU?  ")
# print(result)
# print(["hello", "!", "how", "are", "you", "?"])
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

TRAINED_MODDEL_DIR='../model/sst2/'
data_dir = '../data/glue_data/SST-2'
output_dir = '../model/sst2/eval/'
max_seq_length = 128
vocab_file='../data/cased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True
eval_batch_size = 8
learning_rate=2e-5
# init_checkpoint='./data/cased_L-12_H-768_A-12/bert_model.ckpt'
init_checkpoint='../model/sst2/eval/model.ckpt-6313'
# init_checkpoint='../model/sst2/eval/checkpoint'
bert_config_file='../data/cased_L-12_H-768_A-12/bert_config.json'


get_model_size(init_checkpoint)
exit()


processor = SstProcessor()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=do_lower_case)

train_examples = processor.get_train_examples(data_dir)
trn_file = os.path.join(output_dir, "trn.tf_record")
file_based_convert_examples_to_features(
    train_examples, label_list, max_seq_length, tokenizer, trn_file)

eval_examples = processor.get_dev_examples(data_dir)
eval_file = os.path.join(output_dir, "eval.tf_record")
file_based_convert_examples_to_features(
    eval_examples, label_list, max_seq_length, tokenizer, eval_file)

tst_examples = processor.get_test_examples(data_dir)
tst_file = os.path.join(output_dir, "tst.tf_record")
file_based_convert_examples_to_features(tst_examples, label_list,
                                        max_seq_length, tokenizer,
                                        tst_file)

bert_config = modeling.BertConfig.from_json_file(bert_config_file)
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
    cluster=None,
    master=None,
    model_dir=output_dir,
    save_checkpoints_steps=1000,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=1000,
        num_shards=8,
        per_host_input_for_training=is_per_host))

model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
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

tf.logging.info("***** Running evaluation *****")
tf.logging.info("  Num examples = %d", len(eval_examples))
tf.logging.info("  Batch size = %d", eval_batch_size)

# This tells the estimator to run through the entire set.
eval_steps = None
eval_drop_remainder = False
eval_input_fn = file_based_input_fn_builder(
    input_file=eval_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=eval_drop_remainder)


result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
embeddings = estimator.get_variable_value(estimator.get_variable_names()[12])
with open(output_dir+'embedding.cpkl', 'wb') as handle:
    cPickle.dump(embeddings, handle)

result_predict = [val for val in estimator.predict(eval_input_fn)]
output_eval_file = os.path.join(output_dir, "eval_results.txt")
with tf.gfile.GFile(output_eval_file, "w") as writer:
  tf.logging.info("***** Eval results *****")
  for key in sorted(result.keys()):
    tf.logging.info("  %s = %s", key, str(result[key]))
    writer.write("%s = %s\n" % (key, str(result[key])))