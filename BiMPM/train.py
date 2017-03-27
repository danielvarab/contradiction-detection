import argparse
import sys

from preprocessing import *
from model import *
from callbacks import *

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="snli train file. JSONL file plz")
parser.add_argument('--test', required=True, help="snli test file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")

args = parser.parse_args(sys.argv[1:])

embedding = load_embeddings(args.embedding)

test_data, test_max_sentence_length, test_max_word_length, _ = read_corpus(args.test)
# useful_data, max_sentence_length, max_word_length, list(char_vocab)
s1_filename = "data/sentences_1.txt"
s2_filename = "data/sentences_2.txt"
label_filename = "data/labels.txt"
stats_dic = preprocess_corpus(args.train)

max_sentence_length = stats_dic["sentence_length"]
max_word_length = stats_dic["word_length"]
char_vocab = stats_dic["char_vocab"]
sample_count = stats_dic["sample_count"]
embedding_size = len(char_vocab)+1 # plus for because we need to consider zero 0

encoder = generate_char_encoder(char_vocab)

test_data = create_dataset(test_data, embedding, encoder, max_sentence_length, max_word_length)
test_sentences1, test_sentence1_c, test_sentences2, test_sentence2_c, test_labels  = test_data

# assert len(sentences1) == len(sentences2), "sentence count aren't the same"
# assert len(sentences1) == len(labels), "label count don't match sentence count"

model = build_model_2(embedding_size, max_sentence_length, max_word_length)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

subset = 500
test_sentence_input = {
    "word_sentence_A": test_sentences1[:subset],
    "char_sentence_A": test_sentence1_c[:subset],
    "word_sentence_B": test_sentences2[:subset],
    "char_sentence_B": test_sentence2_c[:subset]
}

test_labels = test_labels[:subset]

sample_generator = create_dataset_generator(embedding, encoder, max_sentence_length, max_word_length, batch_size=32)
model.fit_generator(sample_generator, samples_per_epoch=sample_count, nb_epoch=10, validation_data=(test_sentence_input, test_labels))
model.save('BiMPM.h5')
# model.fit(sentence_input, labels, nb_epoch=10, batch_size=32)
global_scores = model.evaluate(sentence_input, labels, verbose=0)
