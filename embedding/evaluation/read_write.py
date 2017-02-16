import sys
import gzip
import numpy
import math

import numpy as np

from collections import Counter
from operator import itemgetter

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):
  word_vecs = {}
  if filename.endswith('.gz'): file_object = gzip.open(filename, 'r')
  else: file_object = open(filename, 'r')

  for line_num, line in enumerate(file_object):
    line = line.strip().lower()
    word = line.split()[0]
    word_vecs[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vec_val in enumerate(line.split()[1:]):
      word_vecs[word][index] = float(vec_val)
    ''' normalize weight vector '''
    word_vecs[word] /= math.sqrt((word_vecs[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return word_vecs

"""
	Info:
		Loads embedding file
	Input:
		emb_file (string):
			name of file that contains the embeddings.
			the formatting of the file is dictionary-like <word> <embedding>.
			(seperated by whitespace. first entry denotes the key)
		normalize (bool):
			argument to wether the vectors should be normalized or not


"""
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

"""
	Info:
		Loads embedding from 2 source files. Note that this is much slower with two files (probably a TODO)
	Input:
		emb_file:
			name of file that contains the embeddings.
			the formatting of the file is dictionary-like <word> <embedding>.
			(seperated by whitespace. first entry denotes the key)
			vocab_file: file containing list of words corresponding to vectors
		normalize:
			argument to wether the vectors should be normalized or not
"""
def load_embedding2(emb_file, vocab_file, normalize=False):
	n_file = open(name_file)
	v_file = open(vector_file)

	words = n_file.readlines()
	vectors = v_file.readlines()

	for index, word in enumerate(words):
		word = word.rstrip()
		vector = np.array(vectors[index].split(), dtype=np.float32)
		if normalize:
			vector /= math.sqrt((vector**2).sum() + 1e-6)
		word_vectors[name] = vector

	word_vectors


''' Load GRE or TOEFLE task of the format word:word_candidates:gold_label '''
def load_toefle(path) :
	task = []
	with open(path, 'r') as ant_file:
		for l in ant_file:
			a = l.split(':')
			task.append((a[0].strip(), a[1].strip().split(' ') ,a[3].strip()))

	return task
