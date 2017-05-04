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
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, Lambda, concatenate, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers.merge import Concatenate, Dot, maximum
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text
from custom_layers import *
# from callbacks import *
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")
parser.add_argument('--agg_we', required=True, help="aggregation operation")

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

VOCAB_SIZE = len(tokenizer.word_counts) + 1
SENTENCE_MAX_LEN = 42
WORD_DIM = 300
EPOCHS = 42
PATIENCE = 4
BATCH_SIZE = 512
DENSE_NEURON_COUNT = 600
DP = 0.2
L2 = 4e-6

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=SENTENCE_MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print("> fetching embedding")
embedding_matrix = get_embedding_matrix(args.embedding, VOCAB_SIZE, WORD_DIM, tokenizer)

embedding = Embedding(VOCAB_SIZE, WORD_DIM, weights=[embedding_matrix], input_length=SENTENCE_MAX_LEN, trainable=False)

translate = TimeDistributed(Dense(WORD_DIM, activation='relu'))

premise = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_a")
hypothesis = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_b")

prem = embedding(premise)
hypo = embedding(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

aggre_operation = Aggregate(operator=args.agg_we, axis=1)
prem = aggre_operation(prem)
hypo = aggre_operation(hypo)

prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

joint = concatenate([prem,hypo])
joint = Dropout(DP)(joint)
for i in range(3):
    joint = Dense(DENSE_NEURON_COUNT, activation='relu', W_regularizer=l2(L2))(joint)
    joint = Dropout(DP)(joint)
    joint = BatchNormalization()(joint)

prediction = Dense(3, activation='softmax')(joint)

model = Model(inputs=[premise,hypothesis], outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print("Training")
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
early_stop = EarlyStopping(patience=PATIENCE)
chech_point = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)
callbacks = [early_stop, chech_point]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks, verbose=1)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
