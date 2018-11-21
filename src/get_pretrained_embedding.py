import os
import argparse
import modeling
import cPickle
import tensorflow as tf
from run_classifier import model_fn_builder


def save_embedding(basepath, bert_type):
    save_checkpoints_steps = 1000
    iterations_per_loop = 1000
    num_tpu_cores = 8
    learning_rate = 5e-5

    tf.logging.set_verbosity(tf.logging.INFO)

    bert_pretrained_dir = '%s/%s' % (basepath, bert_type)
    output_dir = bert_pretrained_dir
    bert_config_file = '%s/bert_config.json' % (bert_pretrained_dir)
    init_checkpoint = '%s/bert_model.ckpt' % (bert_pretrained_dir)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

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

    embeddings = estimator.get_variable_value(estimator.get_variable_names()[4])
    print(output_dir + '/embedding.cpkl')
    with open(output_dir + '/embedding.cpkl', 'wb') as handle:
        cPickle.dump(embeddings, handle)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data')  # base path
    parser.add_argument('-b', default='uncased_L-24_H-1024_A-16',
                        choices=['cased_L-12_H-768_A-12', 'uncased_L-24_H-1024_A-16'])  # bert model type
    parser.add_argument('-g', default='1')  # gpunum
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.g


    save_embedding(args.p, args.b)
