import sys
import os
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
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, Lambda, concatenate, Dropout, add, multiply
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text
from custom_layers import Align, Aggregate, _align, _aggregate
# from callbacks import LossHistory
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")
parser.add_argument('--ant_embedding', required=True, help="antnonym embedding file")
parser.add_argument('--align_op_we', required=False, help='operator used to sqaush sentence alignment matrix')
parser.add_argument('--agg_we', required=False, help='operator for aggregating over sentence embeddings')
parser.add_argument('--align_op_ae', required=False, help='operator used to sqaush antonym alignment matrix')
parser.add_argument('--agg_ae', required=False, help='operator for aggregating over antonym embeddings')
parser.add_argument('--outfile', required=False, help="File to write results in")
parser.add_argument('--neurons', default=300, help='Number of hidden nodes')
parser.add_argument('--patience', default=8, help='How long should I wait because breaking out?')
parser.add_argument('--timedist', required=False, default=True, help='boolean for applying timedistributed layer after embedding')


args = parser.parse_args(sys.argv[1:])

start = timeit.default_timer()

## SETUP LOGGER
VERBOSE = 1
if args.outfile:
    print = logging.info
    VERBOSE = 2
    logging.basicConfig(filename=args.outfile, level=logging.INFO)
    stdLogger=logging.StreamHandler(sys.stdout)
    stdLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))

    logging.getLogger().addHandler(stdLogger)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        print(" First epoch has been initiated")
    def on_epoch_end(self, batch, logs={}):
        # print(str(logs))
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        print(" Epochs {} - acc : {:.4f}, loss: {:.4f}, val_acc: {:.4f}".format(batch, acc, loss, val_acc))

print(' PARAMETERS:')
print(' Embedding file: ' + args.embedding)
print(' Ant Embedding file: {0}'.format(args.ant_embedding))
print(" Word Embedding Alignment OP: {0}".format(args.align_op_we))
print(" Word Embedding Aggregation OP: {0}".format(args.agg_we))
print(" Ant Embedding Alignment OP: {0}".format(args.align_op_ae))
print(" Ant Embedding Aggregation OP: {0}".format(args.agg_ae))

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
    print(" left sentence length in {0}: {1}".format(fn.split("/")[-1], max(len(x.split()) for x in left)))
    print(" right sentence length in {0}: {1}".format(fn.split("/")[-1], max(len(x.split()) for x in right)))

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
SENT_HIDDEN_SIZE = args.neurons
ANT_SENT_HIDDEN_SIZE = 200
BATCH_SIZE = 512
PATIENCE = args.patience
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

prem_reps = [] # premise sentence representations
hypo_reps = [] # hypothesis sentence representations

# read in embedding and translate
if args.agg_we != None or args.align_op_we != None:
    print(" fetching word embedding")
    embedding_matrix = get_embedding_matrix(args.embedding, VOCAB, EMBED_HIDDEN_SIZE, tokenizer)
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=False)

    prem = embed(premise)
    hypo = embed(hypothesis)

    if args.timedist:
        translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

        prem = translate(prem)
        hypo = translate(hypo)

    prem_reps.append(prem)
    hypo_reps.append(hypo)

# read in antonym embedding and translate
if args.agg_ae != None or args.align_op_ae != None:
    print(" fetching antonym word embedding")
    antonym_embedding_matrix = get_embedding_matrix(args.ant_embedding, VOCAB, ANT_SENT_HIDDEN_SIZE, tokenizer)
    antonym_embed = Embedding(VOCAB, ANT_SENT_HIDDEN_SIZE, weights=[antonym_embedding_matrix], input_length=MAX_LEN, trainable=False)

    ant_prem = antonym_embed(premise)
    ant_hypo = antonym_embed(hypothesis)

    if args.timedist:
        antonym_translate = TimeDistributed(Dense(ANT_SENT_HIDDEN_SIZE, activation=ACTIVATION))

        ant_prem = antonym_translate(ant_prem)
        ant_hypo = antonym_translate(ant_hypo)

if args.align_op_we is not None:
    alignment = _align(prem, hypo, normalize=True)
    alignment_aggr_1 = _aggregate(alignment, args.align_op_we, axis=1)
    alignment_aggr_2 = _aggregate(alignment, args.align_op_we, axis=2)
    prem_reps.append(alignment_aggr_1)
    #prem_reps.append(add(prem,-alignment_aggr_1))
    prem_reps.append(multiply(prem, alignment_aggr_1))
    hypo_reps.append(alignment_aggr_2)
    #hypo_reps.append(add(hypo, -alignment_aggr_2))
    hypo_reps.append(multiply(hypo,alignment_aggr_2))

if args.align_op_ae is not None:
    alignment = _align(ant_prem, ant_hypo, normalize=True)
    alignment_aggr_1 = _aggregate(alignment, args.align_op_ae, axis=1)
    alignment_aggr_2 = _aggregate(alignment, args.align_op_ae, axis=2)
    prem_reps.append(multiply(prem, alignment_aggr_1))
    hypo_reps.append(multiply(hypo, alignment_aggr_2))

assert len(prem_reps) > 0, "no sentence representations, hence no output of the translation layer"

prem_rep = concatenate(prem_reps)
hypo_rep = concatenate(hypo_reps)
print(prem_rep.shape())
prem_rep = Dropout(DP)(prem_rep)
prem_rep = Dropout(DP)(hypo_rep)
for i in range(2):
    prem_rep = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2), name='Prem Dense')(prem_rep)
    prem_rep = Dropout(DP)(prem_rep)
    prem_rep = BatchNormalization()(prem_rep)
    hypo_rep = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2), name='Hypo Dense')(hypo_rep)
    hypo_rep = Dropout(DP)(hypo_rep)
    hypo_rep = BatchNormalization()(hypo_rep)

prem_sum = _aggregate(prem_rep, 'SUM', axis=1)
prem_max = _aggregate(prem_rep, 'MAX', axis=1)
hypo_sum = _aggregate(hypo_rep, 'SUM', axis=1)
hypo_max = _aggregate(hypo_rep, 'MAX', axis=1)

joint = concatenate(prem_sum, prem_max, hypo_sum, hypo_max)

joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2), name='Joint Dense')(joint)
joint = Dropout(DP)(joint)
joint = BatchNormalization()(joint)

pred = Dense(3, activation='softmax')(joint)

model = Model(inputs=[premise, hypothesis], outputs=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print(" Training")
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
early_stop = EarlyStopping(patience=PATIENCE)
check_point = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)
loss_history = LossHistory()
callbacks = [early_stop, check_point, loss_history]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks, verbose=VERBOSE)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print(' Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

ops = ""
for op in [args.agg_ae, args.agg_we, args.align_op_ae, args.align_op_we]:
    if op is None: continue
    ops += "_" + op

name = os.path.splitext(os.path.basename(args.embedding))[0]
pred_file = open("pred_{0}_{1}_{2}.txt".format(name, ops, str(args.neurons) + "n"), "w")
y_proba = model.predict([test[0], test[1]])
for pred in y_proba:
    pred = np.argmax(pred)
    if (pred == 0):
        pred_file.write("contradiction" + "\n")
    elif (pred == 1):
        pred_file.write("neutral" + "\n")
    elif (pred == 2):
        pred_file.write("entailment" + "\n")

pred_file.close()

stop = timeit.default_timer()
time = (stop - start)/60

print(" Model trained in {:.2f} mins".format(time))
