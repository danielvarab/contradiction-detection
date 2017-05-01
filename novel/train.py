import sys
import json
import tempfile
import argparse
import logging
import timeit

import numpy as np

from keras import backend as K
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Input, Dense, LSTM, TimeDistributed, Bidirectional, Flatten, Embedding, Lambda, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.layers.merge import Concatenate, Dot, maximum
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text
from custom_layers import *
from preprocess import get_embedding_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="train file. JSONL file plz")
parser.add_argument('--dev', required=True, help="dev file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")
parser.add_argument('--ant_embedding', required=True, help="antnonym embedding file")
parser.add_argument('--timedist', required=False, default=False, help='boolean for applying timedistributed layer after embedding')
parser.add_argument('--align_op_we', required=False, help='operator used to sqaush sentence alignment matrix')
parser.add_argument('--agg_we', required=False, help='operator for aggregating over sentence embeddings')
parser.add_argument('--align_op_ae', required=False, help='operator used to sqaush antonym alignment matrix')
parser.add_argument('--agg_ae', required=False, help='operator for aggregating over antonym embeddings')
parser.add_argument('--neurons', default=300, help='Number of hidden nodes')
parser.add_argument('--patience', default=8, help='How long should I wait because breaking out?')
parser.add_argument('--outfile', required=True, help="File to write results in")

args = parser.parse_args(sys.argv[1:])

train_file = args.train
dev_file = args.dev
test_file = args.test
emb_file = args.embedding
ant_emb_file = args.ant_embedding

start = timeit.default_timer()

## SETUP LOGGER
outfile = args.outfile

logging.basicConfig(filename=outfile, level=logging.INFO)
stdLogger=logging.StreamHandler(sys.stdout)
stdLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))

logging.getLogger().addHandler(stdLogger)

logging.info('PARAMETERS:\n')
logging.info('\nEmbedding file: ' + emb_file)
logging.info('\nAnt Embedding file: ' + ant_emb_file)
logging.info('\nNeurons: ' + args.neurons)
logging.info('\nPatience: ' + args.patience)
logging.info('\nTimedistribution: ' + str(args.timedist))
logging.info('\nOperators : \n')
logging.info("\tWord Embedding Alignment OP: " + str(args.align_op_we) + "\n")
logging.info("\tWord Embedding Aggregation OP: " + str(args.agg_we) + "\n")
logging.info("\tAnt Embedding Alignment OP: " + str(args.align_op_ae) + "\n")
logging.info("\tAnt Embedding Aggregation OP: " + str(args.agg_ae) + "\n")




logging.info('Arguments: ' + str(args) + '\n\n')

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
#https://github.com/fchollet/keras/issues/1072
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]

def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1 for _, s1, s2 in raw_data]
    right = [s2 for _, s1, s2 in raw_data]
    # words
    logging.info(max(len(x.split()) for x in left))
    logging.info(max(len(x.split()) for x in right))

    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

training = get_data(train_file)
validation = get_data(dev_file)
test = get_data(test_file)

#https://github.com/fchollet/keras/issues/1072
#keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

VOCAB_SIZE = len(tokenizer.word_counts) + 1
SENTENCE_MAX_LEN = 42
WORD_DIM = 300
ANT_WORD_DIM = 200
EPOCHS = 100
PERSPECTIVES = 5
CLASS_COUNT = 3
PATIENCE = int(args.patience)
BATCH_SIZE = 32
DENSE_NEURON_COUNT = int(args.neurons)

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=SENTENCE_MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

logging.info("> fetching embedding")
embedding_matrix = get_embedding_matrix(emb_file, VOCAB_SIZE, WORD_DIM, tokenizer)

logging.info("> fetching antonym embedding")
ant_embedding_matrix = get_embedding_matrix(ant_emb_file, VOCAB_SIZE, ANT_WORD_DIM, tokenizer)

embed = Embedding(VOCAB_SIZE, WORD_DIM, weights=[embedding_matrix], input_length=SENTENCE_MAX_LEN, trainable=False)
ant_embed = Embedding(VOCAB_SIZE, ANT_WORD_DIM, weights=[ant_embedding_matrix], input_length=SENTENCE_MAX_LEN, trainable=False)

translate = TimeDistributed(Dense(WORD_DIM, activation='relu'))
ant_translate = TimeDistributed(Dense(ANT_WORD_DIM, activation='relu'))

premise = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_a")
hypothesis = Input(shape=(SENTENCE_MAX_LEN,), dtype='int32', name="sentence_b")

s1 = embed(premise)
s2 = embed(hypothesis)

ant_s1 = ant_embed(premise)
ant_s2 = ant_embed(hypothesis)

if args.timedist:
    logging.info ('Using time distribution')
    s1 = translate(s1)
    s2 = translate(s2)

    ant_s1 = ant_translate(ant_s1)
    ant_s2 = ant_translate(ant_s2)

if args.agg_we == 'SUM':
    aggre_operation = Lambda(lambda x: K.sum(x, axis=1))
elif args.agg_we == 'MAX':
    aggre_operation = Lambda(lambda x: K.max(x, axis=1))
elif args.agg_we == 'MEAN':
    aggre_operation = Lambda(lambda x: K.mean(x, axis=1))
elif args.agg_we is not None:
    logging.info('Invalid operator for WE aggregation')
    exit(1)

if args.agg_ae == 'SUM':
    ant_aggre_operation = Lambda(lambda x: K.sum(x, axis=1))
elif args.agg_ae == 'MAX':
    ant_aggre_operation = Lambda(lambda x: K.max(x, axis=1))
elif args.agg_ae == 'MEAN':
    ant_aggre_operation = Lambda(lambda x: K.mean(x, axis=1))
elif args.agg_ae is not None:
    logging.info('Invalid operator for AE aggregation')
    exit(1)


if args.align_op_we == 'SUM':
    operation_aligner = Lambda(sum_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_we == 'MAX':
    operation_aligner = Lambda(max_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_we == 'MEAN':
    operation_aligner = Lambda(mean_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_we is not None:
    logging.info('Invalid operator for WE alignment')
    exit(1)

if args.align_op_ae == 'SUM':
    ant_operation_aligner = Lambda(sum_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_ae == 'MAX':
    ant_operation_aligner = Lambda(max_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_ae == 'MEAN':
    ant_operation_aligner = Lambda(mean_both_directions, output_shape=both_directions_output_shape)
elif args.align_op_ae is not None:
    logging.info('Invalid operator for AE alignment')
    exit(1)

tensor = []

if args.agg_we is not None:
    logging.info('Using aggregation of WE with operator ' + args.agg_we)
    agg1 = aggre_operation(s1)
    agg2 = aggre_operation(s2)
    if args.agg_we == 'SUM':
        agg1 = BatchNormalization()(agg1)
        agg2 = BatchNormalization()(agg2)
    tensor.append(agg1)
    tensor.append(agg2)

if args.agg_ae is not None:
    logging.info('Using aggregation of AE with operator ' + args.agg_ae)
    ant_agg1 = ant_aggre_operation(s1)
    ant_agg2 = ant_aggre_operation(s2)
    if args.agg_ae == 'SUM':
        ant_agg1 = BatchNormalization()(ant_agg1)
        ant_agg2 = BatchNormalization()(ant_agg2)
    tensor.append(ant_agg1)
    tensor.append(ant_agg2)

if args.align_op_we is not None:
    logging.info('Using alignment of WE with operator ' + args.align_op_we)
    alignment = Align()([s1,s2])
    if args.align_op_we == 'SUM':
        alignment = BatchNormalization()(alignment)
    tensor.append(operation_aligner(alignment))

if args.align_op_ae is not None:
    logging.info('Using alignment of AE with operator ' + args.align_op_ae)
    alignment = Align()([ant_s1,ant_s2])
    if args.align_op_ae == 'SUM':
        alignment = BatchNormalization()(alignment)
    tensor.append(ant_operation_aligner(alignment))

if(len(tensor) == 0):
    logging.info("Invalid length of tensor, quitting!")
    exit(1)
elif(len(tensor) == 1):
    logging.info("list of tensor has only a single element")
    aggregation = tensor[0]
else:
    aggregation = concatenate(tensor, axis=-1)

prediction = Dense(DENSE_NEURON_COUNT, activation='relu')(aggregation)
prediction = Dense(DENSE_NEURON_COUNT, activation='relu')(prediction)
prediction = Dense(DENSE_NEURON_COUNT, activation='relu')(prediction)

prediction = Dense(CLASS_COUNT, activation='softmax')(prediction)

model = Model(inputs=[premise,hypothesis], outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
logging.info('Number of trainable parameters: ' + str(int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))))

class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        print(str(logs))
        loss = logs.get('loss')
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        logging.info("Epoch: " + str(batch) +"/" + str(EPOCHS)  + " ACC: " + str(acc) + " LOSS: " + str(loss) + " VAL_ACC: " + str(val_acc))

logging.info('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True), LossHistory()]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks, verbose=2)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
logging.info('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

logging.info('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc) + '\n')


stop = timeit.default_timer()
time = (stop - start)/60
logging.info("TIME TAKEN : " + str(time) + " mins")