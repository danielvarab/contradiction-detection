import sys
import re
import argparse

import numpy as np
import matplotlib.pyplot as plot

from sklearn.manifold import TSNE

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

def load_embedding(emb_file, normalize=False, to_lower=False):
	word_vectors = {}
	f = open(emb_file)

	for row in f:
		row = row.split()
		word = row[0].rstrip()
		vector = np.array(row[1:], dtype=np.float32)
		if to_lower:
			word = word.lower()
		if normalize:
			vector /= math.sqrt((vector**2).sum() + 1e-6)
		word_vectors[word] = vector

	return word_vectors

def read_lexicon(filename, wordVecs):
	lexicon = {}
	for line in open(filename, 'r'):
		words = line.lower().strip().split()
		lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
	return lexicon

parser = argparse.ArgumentParser()
parser.add_argument('--e', required=True, help="embedding file")
parser.add_argument('--a', required=True, help="antonym file")
parser.add_argument('--s', required=True, help="synonym file")
parser.add_argument('--w', required=True, help="word")

args = parser.parse_args(sys.argv[1:])
word = args.w

print(">> loading embedding")
embedding = load_embedding(args.e)
print(">> embedding loaded")

print(">> loading antonyms")
antonyms = read_lexicon(args.a, embedding)[word]
print(">> antonyms loaded")

print(">> loading synonyms")
synonyms = read_lexicon(args.s, embedding)[word]
print(">> synonyms loaded")

target_color = "green"
synonym_color = "blue"
antonym_color = "red"

vecs = [ embedding[word] ]
labels = [ target_color ]

for synonym in synonyms:
	vec = embedding.get(synonym, None)
	if vec is None: continue
	vecs.append(vec)
	labels.append(synonym_color)

for antonym in antonyms:
	vec = embedding.get(antonym, None)
	if vec is None: continue
	vecs.append(vec)
	labels.append(antonym_color)

# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# model = TSNE(n_components=2)
# np.set_printoptions(suppress=True)
# Y = model.fit_transform(X)

X = np.array(vecs)
model = TSNE(n_components=2)
np.set_printoptions(suppress=True)
Y = model.fit_transform(X)

plot.scatter(Y[:,0], Y[:,1], 20, labels)
plot.show()
