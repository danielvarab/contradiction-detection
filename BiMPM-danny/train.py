import argparse
import sys

from preprocessing import *
from model import *
from callbacks import *

from keras.callbacks import ModelCheckpoint

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="snli train file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")

args = parser.parse_args(sys.argv[1:])

print(">> loading embedding")
embedding = load_embeddings(args.embedding)
print(">> done loading embedding")

print(">> preprocessing corpus")
stats_dic = preprocess_corpus(args.train)
print(">> done preprocessing corpus")

max_sentence_length = stats_dic["sentence_length"]
max_word_length = stats_dic["word_length"]
char_vocab = stats_dic["char_vocab"]
sample_count = stats_dic["sample_count"]
embedding_size = len(char_vocab)+1 # plus for because we need to consider zero 0

print(">> generating char encoder from vocab")
encoder = generate_char_encoder(char_vocab)

print(">> building model from stat")
model = build_model_2(embedding_size, max_sentence_length, max_word_length)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

print(">> building dataset generator")
sample_generator = create_dataset_generator(embedding, encoder, max_sentence_length, max_word_length, batch_size=128)
print(">> done building dataset generator")

save_callback = ModelCheckpoint("trained_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5")
model.fit_generator(sample_generator, samples_per_epoch=sample_count, nb_epoch=10, callbacks=[save_callback])

# model.fit(sentence_input, labels, nb_epoch=10, batch_size=32)
# global_scores = model.evaluate(sentence_input, labels, verbose=0)
