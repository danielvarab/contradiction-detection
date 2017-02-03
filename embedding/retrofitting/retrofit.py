import argparse
import gzip
import math
import numpy
import re
import sys

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
def read_word_vecs(filename):
  wordVectors = {}
  if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
  else: fileObject = open(filename, 'r')

  for line in fileObject:
    line = line.strip().lower()
    word = line.split()[0]
    wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
    for index, vecVal in enumerate(line.split()[1:]):
      wordVectors[word][index] = float(vecVal)
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)

  sys.stderr.write("Vectors read from: "+filename+" \n")
  return wordVectors

''' Read all the word vectors '''
def load_embedding_from_two_files(name_file, vector_file):
  with open(name_file, "r") as n_file, open(vector_file) as v_file:
    names = n_file.readlines()
    vectors = v_file.readlines()

    dic = {}
    for index, name in enumerate(names):
      row = vectors[index].split()
      dic[name.rstrip()] = np.array(row).astype(float)
    return dic

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

        new_words[word] = new_vector/((alpha+beta*len(synonym_neighbours)+gamma*len(synonym_neighbours)))

    return new_words


if __name__=='__main__':

  name = "glove.6B.300d"
  wordVecs = read_word_vecs("../../datasets/glove.6b/"+name+".txt")
  synonyms = read_lexicon("lexicons/synonym.txt", wordVecs)
  antonyms = read_lexicon("lexicons/antonym.txt", wordVecs)
  #ppdb = read_lexicon("lexicons/ppdb-xl.txt", wordVecs)
  #wnsyn = read_lexicon("lexicons/wordnet-synonyms.txt", wordVecs)
  #wnall = read_lexicon("lexicons/wordnet-synonyms+.txt", wordVecs)
  #fn = read_lexicon("lexicons/framenet.txt", wordVecs)
  numIter = 10
  #outFileName1 = "{}{}".format(name,"_ppdb_out.txt")
  #outFileName2 = "{}{}".format(name,"_wnsyn_out.txt")
  #outFileName3 = "{}{}".format(name,"_wnall_out.txt")
  #outFileName4 = "{}{}".format(name,"_fn_out.txt")
  outFileName5 = "{}{}".format(name,"_new_anto_rf_out.txt")

  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  #print_word_vecs(retrofit(wordVecs, ppdb, numIter), outFileName1)
  #print_word_vecs(retrofit(wordVecs, wnsyn, numIter), outFileName2)
  #print_word_vecs(retrofit(wordVecs, wnall, numIter), outFileName3)
  #print_word_vecs(retrofit(wordVecs, fn, numIter), outFileName4)
  print_word_vecs(retrofit_v2(wordVecs, synonyms, antonyms, numIter), outFileName5)











