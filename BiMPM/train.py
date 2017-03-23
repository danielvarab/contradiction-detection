import argparse
import sys

from preprocessing import *
from model import build_model

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="snli train file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")

args = parser.parse_args(sys.argv[1:])

embedding = load_embeddings(args.embedding)

training_data, max_sentence_length, max_word_length, char_vocab = read_corpus(args.train)

encoder = generate_char_encoder(char_vocab)

training_data = create_dataset(training_data, embedding, encoder, max_sentence_length, max_word_length)
sentences1, sentence1_c, sentences2, sentence2_c, labels  = training_data

assert len(sentences1) == len(sentences2), "sentence count aren't the same"
assert len(sentences1) == len(labels), "label count don't match sentence count"

embedding_size = len(char_vocab)+1 # plus for because we need to consider zero 0
model = build_model(embedding_size, max_sentence_length, max_word_length)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


sample_count = len(sentences1)
sentence_input = {
    "word_sentence_A": sentences1,
    "char_sentence_A": sentence1_c,
    "word_sentence_B": sentences2,
    "char_sentence_B": sentence2_c,
}

labels = np.random.randint(2, size=sample_count)
model.fit(sentence_input, labels, nb_epoch=10, batch_size=32)
