import argparse
import gzip
import math
import numpy
import re
import sys

from copy import deepcopy
from autograd import grad
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

        alpha = len(synonym_neighbours)
        if alpha is 0: alpha = 1
        beta  = 1
        gamma = 1

        new_vector = alpha * words[word] # origin cost

        for synonym in synonym_neighbours: # synonym cost
            new_vector += (beta * new_words[synonym])

        for antonym in antonym_neighbours: # antonym cost
            new_vector -= (gamma * new_words[antonym])

        new_words[word] = new_words - new_vector/(2*(alpha+beta+gamma))

    return new_words


if __name__=='__main__':

  wordVecs = read_word_vecs("sample_vec.txt")
  synonyms = read_lexicon("lexicons/synonym.txt", wordVecs)
  antonyms = read_lexicon("lexicons/antonym.txt", wordVecs)
  numIter = 10
  outFileName = "out_vec_1.txt"

  ''' Enrich the word vectors using ppdb and print the enriched vectors '''
  # print_word_vecs(retrofit(wordVecs, lexicon, numIter), outFileName)
  print_word_vecs(retrofit_v2(wordVecs, synonyms, antonyms, numIter), outFileName)
