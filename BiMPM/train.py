import argparse
import sys

from preprocessing import create_dataset
from model import build_model

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help="snli train file. JSONL file plz")
parser.add_argument('--embedding', required=True, help="embedding file")

args = parser.parse_args(sys.argv[1:])

sentences1, sentences2, labels, max_sentence_length = create_dataset(args.train, args.embedding)

assert len(sentences1) == len(sentences2), " count aren't the same"
assert len(sentences1) == len(labels), "label count don't match sentence count"

WORD_LENGTH = 15

model = build_model(max_sentence_length, WORD_LENGTH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


def ri(shape): return np.random.randint(27, size=shape)
sample_count = len(sentences1)
inpuyt = {
    "word_sentence_A": sentences1,
    "char_sentence_A": ri((sample_count, max_sentence_length, 15)),
    "word_sentence_B": sentences2,
    "char_sentence_B": ri((sample_count, max_sentence_length, 15)),
}

labels = np.random.randint(2, size=sample_count)
model.fit(inpuyt, labels, nb_epoch=10, batch_size=32)
