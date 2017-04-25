import sys
import json
import tempfile
import argparse

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, concatenate, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import Concatenate, Dot, maximum

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


from custom_layers import *
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")
parser.add_argument('--ant_embedding', required=True, help="antnonym embedding file")

args = parser.parse_args(sys.argv[1:])

train_file = args.train
dev_file = args.dev
test_file = args.test
emb_file = args.embedding
ant_emb_file = args.ant_embedding

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
    # words
    print(max(len(x.split()) for x in left))
    print(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

training = get_data(train_file)
validation = get_data(dev_file)
test = get_data(test_file)

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

VOCAB_SIZE = len(tokenizer.word_counts) + 1
SENTENCE_MAX_LEN = 42
WORD_DIM = 300
ANT_WORD_DIM = 200
PERSPECTIVES = 5
CLASS_COUNT = 3
PATIENCE = 4
BATCH_SIZE = 32
DENSE_NEURON_COUNT = 200

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=SENTENCE_MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print("> fetching embedding")
embedding_matrix = get_embedding_matrix(emb_file, VOCAB_SIZE, WORD_DIM, tokenizer)

print("> fetching antonym embedding")
ant_embedding_matrix = get_embedding_matrix(ant_emb_file, VOCAB_SIZE, ANT_WORD_DIM, tokenizer)

embed = Embedding(VOCAB_SIZE, WORD_DIM, weights=[embedding_matrix], input_length=SENTENCE_MAX_LEN, trainable=False)
ant_embed = Embedding(VOCAB_SIZE, ANT_WORD_DIM, weights=[ant_embedding_matrix], input_length=SENTENCE_MAX_LEN, trainable=False)

premise = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_a")
hypothesis = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_b")

summer = Lambda(lambda x: K.sum(x, axis=1))
max_aligner = Lambda(max_both_directions, output_shape=max_both_directions_output_shape)

s1 = embed(premise)
s2 = embed(hypothesis)

ant_s1 = ant_embed(premise)
ant_s2 = ant_embed(hypothesis)

alignment = Align()([s1,s2])

max_alignment = max_aligner(alignment)

tensors = [
    summer(s1),
    summer(s2),
    summer(ant_s1),
    summer(ant_s2),
    max_alignment
]

aggregation = concatenate(tensors, axis=-1)

prediction = Dense(DENSE_NEURON_COUNT)(aggregation)
prediction = Dense(DENSE_NEURON_COUNT)(prediction)
prediction = Dense(DENSE_NEURON_COUNT)(prediction)

prediction = Dense(CLASS_COUNT, activation='softmax')(prediction)

model = Model(inputs=[premise,hypothesis], outputs=prediction)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=100, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
