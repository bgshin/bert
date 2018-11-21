import os
import tensorflow as tf
import collections
import tokenization
from run_classifier import SstProcessor, file_based_input_fn_builder, model_fn_builder, file_based_convert_examples_to_features
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.models import Model
import numpy as np

import cPickle

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


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

vocab_file='../data/cased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True
data_dir = '../data/glue_data/SST-2'
max_seq_length = 128
output_dir = '../model/sst2/eval/'
embedding_dir = '../model/sst2/eval/embedding.cpkl'

def _parse_function(proto):
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


def create_dataset(filepath, batch):
    NUM_CLASSES = 2
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    if filepath.endswith('trn.tf_record'):
        dataset = dataset.repeat()

    # Set the batchsize
    dataset = dataset.batch(batch)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # # Bring your picture back in shape
    # image = tf.reshape(image, [-1, 256, 256, 1])

    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label

trn_file = os.path.join(output_dir, "trn.tf_record")
dev_file = os.path.join(output_dir, "eval.tf_record")
tst_file = os.path.join(output_dir, "tst.tf_record")
with open(embedding_dir, 'rb') as handle:
    embeddings = cPickle.load(handle)

#Get your datatensors
tokens_trn, label_trn = create_dataset(trn_file, batch=30)
tokens_dev, label_dev = create_dataset(dev_file, batch=1)
tokens_tst, _ = create_dataset(tst_file, batch=1)

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
# exit()
EPOCHS=5
SUM_OF_ALL_DATASAMPLES = 67350
BATCHSIZE = 400
STEPS_PER_EPOCH = SUM_OF_ALL_DATASAMPLES / BATCHSIZE

model.fit(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
print(1)
print(model.evaluate(tokens_dev, label_dev, steps=872))
result = model.predict(tokens_tst, steps=1820)
print(result.shape)
print(result)
filename = os.path.join(data_dir, "tst.distill.%s.materials.cpkl" % 'cased_L-12_H-768_A-12')
print('tst %s' % filename)
with open(filename, 'rb') as handle:
    (x_tst, y_tst, ysm_tst) = cPickle.load(handle)

yh = np.array([0 if a[0] > a[1] else 1 for a in result])
print(yh.shape)
print(yh)
print(y_tst)
print(np.mean(y_tst==yh))
