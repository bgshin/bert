import os
import tensorflow as tf
import argparse
import tokenization
from run_classifier import SstProcessor, file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.models import Model
import cPickle

def _parse_function(proto, max_seq_length=128):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }
    # 'image': tf.FixedLenFeature([], tf.string),
    #                     "label": tf.FixedLenFeature([], tf.int64)}

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    return parsed_features['input_ids'], parsed_features["label_ids"]


def create_dataset(filepath):
    NUM_CLASSES = 2
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the batchsize
    dataset = dataset.batch(30)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # # Bring your picture back in shape
    # image = tf.reshape(image, [-1, 256, 256, 1])

    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

def train(basepath, task, bert_type):
    max_seq_length = 128
    data_dir = '%s/glue_data/%s' % (basepath, task)
    output_dir = '%s/bert/%s/' % (data_dir, bert_type)
    vocab_file = '%s/%s/vocab.txt' % (basepath, bert_type)
    do_lower_case = True if bert_type.split('_')[0] == 'uncased' else False
    # output_dir = '../model/sst2/eval/'
    embedding_dir = '%s/%s/embedding.cpkl' % (basepath, bert_type)
    trn_file = os.path.join(data_dir, "trn.tf_record")
    dev_file = os.path.join(data_dir, "dev.tf_record")
    tst_file = os.path.join(data_dir, "tst.tf_record")
    with open(embedding_dir, 'rb') as handle:
        embeddings = cPickle.load(handle)

    # Get your datatensors
    tokens_trn, label_trn = create_dataset(trn_file)
    tokens_dev, label_dev = create_dataset(dev_file)

    model_input = keras.layers.Input(tensor=tokens_trn)
    print(model_input)
    z = Embedding(embeddings.shape[0],
                  embeddings.shape[1],
                  weights=[embeddings],
                  input_length=max_seq_length,
                  trainable=False)(model_input)
    filter_sizes = (2, 3, 4, 5)
    num_filters = 64
    dropout_prob = 0.2
    hidden_dims = 50
    batch_size = 64
    epochs = 200

    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv1D(num_filters,
                      sz,
                      padding="valid",
                      activation="relu",
                      strides=1)(z)
        print(conv)
        # conv = MaxPooling1D(pool_size=2)(conv)
        conv = MaxPooling1D(pool_size=max_seq_length - sz + 1)(conv)
        print(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)

    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob)(z)
    model_output = Dense(2, activation="softmax")(z)

    model = Model(model_input, model_output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"], target_tensors=[label_trn])
    model.summary()
    # 22,957,826 ~ 23M
    # 22,268,928 ~ 22M
    # if 50dim -> 324000/(22000*50/768)=x226 compression!

    EPOCHS = 50
    SUM_OF_ALL_DATASAMPLES = 67350
    BATCHSIZE = 2000
    STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / BATCHSIZE

    # checkpoint = MyCallback(data)
    # callbacks_list = [checkpoint]
    # model.fit(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks_list)
    model.fit(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    print(1)
    print(model.evaluate(tokens_dev, label_dev, steps=STEPS_PER_EPOCH))



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

    train(args.p, args.t, args.b)









