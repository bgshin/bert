import os
import argparse
import modeling
import tokenization
import numpy as np
import tensorflow as tf
from run_classifier import ColaProcessor, SstProcessor, MrpcProcessor, QqpProcessor, QnliProcessor, MnliProcessor, RteProcessor, SnliProcessor, WnliProcessor
from run_classifier import file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features


def run_classifier(basepath, task, bert_type, max_seq_length=128):
    # save_checkpoints_steps = 1000
    save_checkpoints_steps = 100
    iterations_per_loop = 1000
    num_tpu_cores = 8
    # train_batch_size = 32
    # eval_batch_size = 8
    # predict_batch_size = 8
    train_batch_size = 6
    eval_batch_size = 6
    predict_batch_size = 6
    learning_rate = 2e-5
    num_train_epochs = 3.0
    warmup_proportion = 0.1

    tf.logging.set_verbosity(tf.logging.INFO)
    data_dir = '%s/glue_data/%s' % (basepath, task)
    output_dir = '%s/bert/%s/' % (data_dir, bert_type)
    vocab_file = '%s/%s/vocab.txt' % (basepath, bert_type)
    bert_pretrained_dir = '%s/%s' % (basepath, bert_type)
    bert_config_file = '%s/bert_config.json' % (bert_pretrained_dir)
    init_checkpoint = '%s/bert_model.ckpt' % (bert_pretrained_dir)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    # choices=['CoLA', 'SST-2', 'MRPC', 'QQP', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'SNLI', 'STS-B'])  # GLUE task type
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

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(output_dir)
    tf.gfile.Copy(vocab_file, output_dir+'vocab.txt', overwrite=True)

    task_name = task.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    # train
    train_examples = processor.get_train_examples(data_dir)
    num_train_steps = int(
        len(train_examples) / train_batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=False,
        use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    # train
    do_lower_case = True if bert_type.split('_')[0] == 'uncased' else False
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    train_file = os.path.join(data_dir, "trn_bert.%s.tf_record" % bert_type)
    # file_based_convert_examples_to_features(
    #     train_examples, label_list, max_seq_length, tokenizer, train_file)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # dev
    dev_examples = processor.get_dev_examples(data_dir)
    eval_file = os.path.join(data_dir, "dev_bert.%s.tf_record" % bert_type)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(dev_examples))
    tf.logging.info("  Batch size = %d", eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    use_tpu = False
    eval_drop_remainder = True if use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    # prediction on test dataset
    predict_examples = processor.get_test_examples(data_dir)
    predict_file = os.path.join(data_dir, "tst_bert.%s.tf_record" % bert_type)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", predict_batch_size)

    predict_drop_remainder = True if use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in result:
            output_line = "\t".join(
                str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)

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

    run_classifier(args.p, args.t, args.b)
