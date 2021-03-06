import sys
import json
import tempfile
import argparse
import logging
import timeit

import numpy as np
np.random.seed(1337)

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
import keras.preprocessing.text
from custom_layers import Align, Aggregate, _align, _aggregate
# from callbacks import *
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")

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
# LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
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

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

# read in embedding and translate
print("> fetching word embedding")
embedding_matrix = get_embedding_matrix(args.embedding, VOCAB, EMBED_HIDDEN_SIZE, tokenizer)
embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)

prem = embed(premise)
hypo = embed(hypothesis)

translate_1 = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))
translate_2 = TimeDistributed(Dense(200, activation=ACTIVATION))

t_prem = translate_1(prem)
t_hypo = translate_1(hypo)

t_prem = _aggregate(t_prem, "SUM", axis=1)
t_hypo = _aggregate(t_hypo, "SUM", axis=1)

a_prem = translate_2(prem)
a_hypo = translate_2(hypo)

align = Align(normalize=True)([a_prem, a_hypo])
print(align.get_shape())

conv1 = Conv1D(32, kernel_size=3, activation='relu')(align)
conv2 = Conv1D(64, 3, activation='relu')(conv1)
pool = MaxPooling1D(pool_size=2)(conv2)
flattened = Flatten()(pool)

# assert len(reps) > 0, "no sentence representations, hence no output of the translation layer"

joint = concatenate([t_prem, t_hypo, flattened])
joint = Dropout(DP)(joint)
for i in range(3):
    joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(joint)
    joint = Dropout(DP)(joint)
    joint = BatchNormalization()(joint)

pred = Dense(3, activation='softmax')(joint)

model = Model(inputs=[premise, hypothesis], outputs=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

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
