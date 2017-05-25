import sys
import json
import tempfile
import argparse
import logging
import timeit

import numpy as np
np.random.seed(1337)

import keras.preprocessing.text

from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, Lambda, concatenate, Dropout, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers.pooling import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.engine.topology import Layer
from custom_layers import Align, Aggregate, _align, _aggregate, _softalign
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")
parser.add_argument('--ant_embedding', required=True, help="antonym embedding file")

args = parser.parse_args(sys.argv[1:])

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit: break

        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))

        if skip_no_majority and label == '-': continue

        yield (label, s1, s2)


def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    print(max(len(x.split()) for x in left))
    print(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

training = get_data(args.train)
validation = get_data(args.dev)
test = get_data(args.test)

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
ANT_SENT_HIDDEN_SIZE = 200
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

class Perspective(Layer):
    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        super(Perspective, self).__init__(**kwargs)

    def build(self, input_shape):
        a_shape = input_shape[0]
        b_shape = input_shape[1]
        self.kernel = self.add_weight(shape=(a_shape[1], a_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        super(Perspective, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a, b = x

        a = a * self.kernel
        b = b * self.kernel

        # if we normalize, this layer outputs the cosine similarity
        if self.normalize:
            a = K.l2_normalize(a, axis=2)
            b = K.l2_normalize(b, axis=2)

        return K.batch_dot(a, b, axes=[2, 2])

    def compute_output_shape(self, input_shape):
        a_shape, b_shape = input_shape
        return (a_shape[0], a_shape[1], b_shape[1])

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

# read in embedding and translate
print("> fetching word embedding")
embedding_matrix = get_embedding_matrix(args.embedding, VOCAB, 300, tokenizer)
embed = Embedding(VOCAB, 300, weights=[embedding_matrix], input_length=42, trainable=False)

prem = embed(premise)
hypo = embed(hypothesis)

translate = TimeDistributed(Dense(300, activation="relu"))

hypo = translate(prem)
prem = translate(hypo)

perspectives = 5
prem_contexts = []
hypo_contexts = []
for p in range(perspectives):
    alignment = Perspective(normalize=True)([prem,hypo])
    prem_c = _softalign(prem, alignment, transpose=True)
    hypo_c = _softalign(hypo, alignment)
    prem_contexts.append(prem_c)
    hypo_contexts.append(hypo_c)

multiplier = Lambda(lambda x: x[0] * x[1])

weighted_prems = [multiplier([prem, hypo_c]) for prem_c in prem_contexts]
weighted_hypos = [multiplier([hypo, prem_c]) for hypo_c in hypo_contexts]

prem = [prem] + weighted_prems
hypo = [hypo] + weighted_hypos

prem = concatenate(prem, axis=-1)
hypo = concatenate(hypo, axis=-1)

project = TimeDistributed(Dense(600, activation="relu"))

prem = project(prem)
hypo = project(hypo)

prem = _aggregate(prem, "SUM", axis=1)
hypo = _aggregate(hypo, "SUM", axis=1)

joint = concatenate([prem, hypo])
joint = Dropout(DP)(joint)
for i in range(3):
    joint = Dense(600, activation="relu", kernel_regularizer=l2(4e-6))(joint)
    joint = Dropout(0.2)(joint)
    joint = BatchNormalization()(joint)

pred = Dense(3, activation='softmax')(joint)

model = Model(inputs=[premise, hypothesis], outputs=pred)
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("Training")
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
early_stop = EarlyStopping(patience=PATIENCE)
chech_point = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)
callbacks = [early_stop, chech_point]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks, verbose=1)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
