import argparse
import gzip
import math
import numpy
import re
import sys
import os

from copy import deepcopy
from scipy.spatial import distance
import numpy as np

isNumber = re.compile(r'\d+.*')
def norm_word(word):
  if isNumber.search(word.lower()):
    return '---num---'
  elif re.sub(r'\W+', '', word) == '':
    return '---punc---'
  else:
    return word.lower()

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename, normalize):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')

  for line in fileObject:
    if line[0].isupper(): continue;
    line = line.strip().lower() # NOTE: Daniel: this reduces the word embedding
    word = line.split()[0]
    vector = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      vector[index] = float(vecVal)

    ''' normalize weight vector '''
    if normalize:
        vector /= math.sqrt((vector**2).sum() + 1e-6)

    wordVectors[word] = vector

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors


''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
  sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
  outFile = open(outFileName, 'w')
  for word, values in wordVectors.iteritems():
    outFile.write(word+' ')
    for val in wordVectors[word]:
      outFile.write('%.4f' %(val)+' ')
    outFile.write('\n')
  outFile.close()

''' Read the PPDB word relations as a dictionary '''
def read_lexicon(filename, wordVecs):
  lexicon = {}
  for line in open(filename, 'r'):
    words = line.lower().strip().split()
    lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
  return lexicon

''' Retrofit word vectors to a lexicon '''
def retrofit(wordVecs, lexicon, numIters):
  newWordVecs = deepcopy(wordVecs)
  wvVocab = set(newWordVecs.keys())
  loopVocab = wvVocab.intersection(set(lexicon.keys()))
  for it in range(numIters):
    # loop through every node also in ontology (else just use data estimate)
    for word in loopVocab:
      wordNeighbours = set(lexicon[word]).intersection(wvVocab)
      numNeighbours = len(wordNeighbours)
      #no neighbours, pass - use data estimate
      if numNeighbours == 0:
        continue
      # the weight of the data estimate if the number of neighbours
      newVec = numNeighbours * wordVecs[word] # produces vector multiplied by of numNeighbours
      # loop over neighbours and add to new vector (currently with weight 1)
      for ppWord in wordNeighbours:
        newVec += newWordVecs[ppWord] # danny: adding two vectors
      newWordVecs[word] = newVec/(2*numNeighbours)
  return newWordVecs


def retrofit_v2(words, synonyms, antonyms, iterations):
    new_words = deepcopy(words)
    vocab = set(words.keys())

    for i in range(iterations):
      for word in vocab:
        synonym_neighbours = set(synonyms.get(word, [])).intersection(vocab)
        antonym_neighbours = set(antonyms.get(word, [])).intersection(vocab)

        alpha = len(synonym_neighbours)+len(antonym_neighbours)
        if alpha is 0: alpha = 1
        #alpha = alpha
        beta  = 1
        gamma = 1

        orig_vector = words[word]
        last_vector = new_words[word]
        v_sum = last_vector + orig_vector
        v_diff = last_vector - orig_vector
        if (np.sqrt(v_diff.dot(v_diff))>np.sqrt(v_sum.dot(v_sum))):
          orig_vector = -orig_vector

        new_vector = alpha * orig_vector # origin cost

        for synonym in synonym_neighbours: # synonym cost
            new_vector += (beta * new_words[synonym])

        for antonym in antonym_neighbours: # antonym cost
            new_vector -= (gamma * new_words[antonym])

        new_words[word] = new_vector/((alpha+beta*len(synonym_neighbours)+gamma*len(antonym_neighbours)))

    return new_words

def rel_path(path):
	return os.path.join(os.path.dirname(__file__), path)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', required=True, help="embedding file")
	parser.add_argument("--normalize", action="store_true", default=False, help="normalize embedding. defaults to False")
	parser.add_argument('--ppdb', action="store_true", default=False, help="retrofit with ppdb")
	parser.add_argument('--wnsyn', action="store_true", default=False, help="retrofit with word net synonyms")
	parser.add_argument('--wnall', action="store_true", default=False, help="retrofit with word net all")
	parser.add_argument('--fn', action="store_true", default=False, help="retrofit with frame net")
	parser.add_argument('--a_s_rf', action="store_true", default=False, help="modified antonym + synonym retrofit")
	args = parser.parse_args(sys.argv[1:])

	name = args.e
	wordVecs = read_word_vecs(args.e, args.normalize)

	numIter = 10

	''' Enrich the word vectors using ppdb and print the enriched vectors '''
	if args.ppdb:
		ppdb_outfile = "{}{}".format(name, "_ppdb_out.txt")
		ppdb_lex_path = rel_path("lexicons/ppdb-xl.txt")
		ppdb = read_lexicon(ppdb_lex_path, wordVecs)
		new_emb = retrofit(wordVecs, ppdb, numIter)
		print_word_vecs(new_emb, ppdb_outfile)

	if args.wnsyn:
		wnsyn_outfile = "{}{}".format(name, "_wnsyn_out.txt")
		wn_lex_path = rel_path("lexicons/wordnet-synonyms.txt")
		wnsyn = read_lexicon(wn_lex_path, wordVecs)
		new_emb = retrofit(wordVecs, wnsyn, numIter)
		print_word_vecs(new_emb, wnsyn_outfile)

	if args.wnall:
		wnall_outfile = "{}{}".format(name, "_wnall_out.txt")
		wn_all_lex_path = rel_path("lexicons/wordnet-synonyms+.txt")
		wnall = read_lexicon(wn_all_lex_path, wordVecs)
		new_emb = retrofit(wordVecs, wnall, numIter)
		print_word_vecs(new_emb, wnall_outfile)

	if args.fn:
		fn_outfile = "{}{}".format(name, "_fn_out.txt")
		lexicon_path = rel_path("lexicons/framenet.txt")
		fn = read_lexicon(lexicon_path, wordVecs)
		new_emb = retrofit(wordVecs, fn, numIter)
		print_word_vecs(new_emb, fn_outfile)

	if args.a_s_rf:
		new_retrofit_outfile = "{}{}".format(name, "_anto_rf_out.txt")
		syn_lex_path = rel_path("lexicons/synonym.txt")
		ant_lex_path = rel_path("lexicons/antonym.txt")
		synonyms = read_lexicon(syn_lex_path, wordVecs)
		antonyms = read_lexicon(ant_lex_path, wordVecs)
		new_emb = retrofit_v2(wordVecs, synonyms, antonyms, numIter)
		print_word_vecs(new_emb, new_retrofit_outfile)
