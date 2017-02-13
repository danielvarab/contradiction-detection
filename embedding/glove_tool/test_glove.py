import logging

from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose

import evaluate
import glove

logger = logging.getLogger("glove")

#def load_model(path, vocab):
	

#def train_small_testmodel():

# Mock corpus (shamelessly stolen from Gensim word2vec tests)

test_corpus = ("""human interface computer
 survey user computer system response time
 eps user interface system
 system human system eps
 user response time
 trees
 graph trees
 graph minors trees
 graph minors survey
 I like graph and stuff
 I like trees and stuff
 Sometimes I build a graph
 Sometimes I build trees""").split("\n")



def read_lines(path):
    with open(path) as f:
        return f.read().split("\n")

print("Loading corpus and lexica")

#test_corpus = read_lines("../../datasets/snli_1.0/snli_sentenceA_72k_train.txt")

synonyms = read_lines("../../datasets/antonym_synonym/synonym_200.txt")
antonyms = read_lines("../../datasets/antonym_synonym/antonym_200.txt")

glove.logger.setLevel(logging.ERROR)

print("Building Vocab")

vocab = glove.build_vocab(test_corpus, synonyms, antonyms)

synonyms = glove.build_syncab(synonyms, vocab)
antonyms = glove.build_antcab(antonyms, vocab)

print("Building Cooccur")

cooccur = glove.build_cooccur(vocab, test_corpus, window_size=1)
id2word = evaluate.make_id2word(vocab)

print("Training vectors...")

W = glove.train_glove(vocab, synonyms, antonyms, cooccur, vector_size=100, iterations=25)

print("Evaluation:")
# Merge and normalize word vectors
W = evaluate.merge_main_context(W)



def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'trees')
    print(similar)
    logger.info(similar)

    #assert_equal('trees', similar[0])

def test_dissimilarity():
    dissimilar = evaluate.least_similar(W, vocab, id2word, 'trees')
    print(dissimilar)
    logger.info(dissimilar)

    #assert_equal('trees', similar[0])


def test_subcost(w1,w2):
    result = evaluate.distance(W, vocab, w1, w2)
    print(result)
    #assert_equal(0, result)

test_similarity()
test_dissimilarity()
#test_subcost('boy','woman')
#test_subcost('boy','walking')
#test_subcost('tall','small')
#test_subcost('small','tall')
#test_subcost('boy','father')



