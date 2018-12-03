import os
import argparse
import modeling
import tokenization
import numpy as np
import tensorflow as tf
from run_classifier import ColaProcessor, SstProcessor, MrpcProcessor, QqpProcessor, QnliProcessor, MnliProcessor, RteProcessor, SnliProcessor, WnliProcessor
from run_classifier import file_based_input_fn_builder, file_based_convert_examples_to_features
import optimization

# def bootstrap(save_file):
#     """Initialize a tf.Estimator run with random initial weights."""
#     # a bit hacky - forge an initial checkpoint with the name that subsequent
#     # Estimator runs will expect to find.
#     #
#     # Estimator will do this automatically when you call train(), but calling
#     # train() requires data, and I didn't feel like creating training data in
#     # order to run the full train pipeline for 1 step.
#
#     sess = tf.Session(graph=tf.Graph())
#     with sess.graph.as_default():
#         features, labels = get_inference_input()
#         model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
#                  params=FLAGS.flag_values_dict())
#         sess.run(tf.global_variables_initializer())
#         tf.train.Saver().save(sess, save_file)


def create_model_distill(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertDistillModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)




def model_fn_builder_distill(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model_distill(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    # scaffold_fn = None
    #### modified ###
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(
        include=['bert/embeddings/word_embeddings:0'])
    init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
        init_checkpoint, variables_to_restore)

    def InitAssignFn(scaffold, sess):
        # print('=================***************** sess.run(tf.global_variables_initializer())================= ')
        # sess.run(tf.global_variables_initializer())
        print('=================***************** sess.run(init_assign_op, init_feed_dict)================= ')
        sess.run(init_assign_op, init_feed_dict)

    def scaffold_fn():
        print('=================***************** scaffold_fn *****************================= ')
        return tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)
    #### modified ###

    initialized_variable_names = []

    #### modified ###
    # if init_checkpoint:
    #   (assignment_map, initialized_variable_names
    #   ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    #   if use_tpu:
    #
    #     def tpu_scaffold():
    #       tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    #       return tf.train.Scaffold()
    #
    #     scaffold_fn = tpu_scaffold
    #   else:
    #     # tf.initializers.global_variables()
    #     tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # if init_checkpoint:
    #   tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    #### modified ###

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=probabilities, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


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
    distill_output_dir = '%s/bert/%s/distill/' % (data_dir, bert_type)
    vocab_file = '%s/%s/vocab.txt' % (basepath, bert_type)
    bert_pretrained_dir = '%s/%s' % (basepath, bert_type)
    bert_config_file = '%s/bert_config.json' % (bert_pretrained_dir)
    init_checkpoint = '%s/bert_model.ckpt' % (bert_pretrained_dir)
    # init_checkpoint = None
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
    tf.gfile.MakeDirs(distill_output_dir)
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
        model_dir=distill_output_dir,
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

    model_fn = model_fn_builder_distill(
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
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    run_classifier(args.p, args.t, args.b)
