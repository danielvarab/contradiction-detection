#!/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import cPickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse
from scipy.special import expit

from util import listify


logger = logging.getLogger("glove")


def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))

    parser.add_argument('corpus', metavar='corpus_path',
                        type=partial(codecs.open, encoding='utf-8'))

    parser.add_argument('synonyms', metavar='synonyms_path',
                        type=partial(codecs.open, encoding='utf-8'))

    parser.add_argument('antonyms', metavar='antonyms_path',
                        type=partial(codecs.open, encoding='utf-8'))

    g_vocab = parser.add_argument_group('Vocabulary options')
    g_vocab.add_argument('--vocab-path',
                         help=('Path to vocabulary file. If this path '
                               'exists, the vocabulary will be loaded '
                               'from the file. If it does not exist, '
                               'the vocabulary will be written to this '
                               'file.'))

    g_cooccur = parser.add_argument_group('Cooccurrence tracking options')
    g_cooccur.add_argument('--cooccur-path',
                           help=('Path to cooccurrence matrix file. If '
                                 'this path exists, the matrix will be '
                                 'loaded from the file. If it does not '
                                 'exist, the matrix will be written to '
                                 'this file.'))
    g_cooccur.add_argument('-w', '--window-size', type=int, default=10,
                           help=('Number of context words to track to '
                                 'left and right of each word'))
    g_cooccur.add_argument('--min-count', type=int, default=10,
                           help=('Discard cooccurrence pairs where at '
                                 'least one of the words occurs fewer '
                                 'than this many times in the training '
                                 'corpus'))

    g_glove = parser.add_argument_group('GloVe options')
    g_glove.add_argument('--vector-path',
                         help=('Path to which to save computed word '
                               'vectors'))
    g_glove.add_argument('-s', '--vector-size', type=int, default=100,
                         help=('Dimensionality of output word vectors'))
    g_glove.add_argument('--iterations', type=int, default=25,
                         help='Number of training iterations')
    g_glove.add_argument('--learning-rate', type=float, default=0.05,
                         help='Initial learning rate')
    g_glove.add_argument('--save-often', action='store_true', default=False,
                         help=('Save vectors after every training '
                               'iteration'))

    return parser.parse_args()


def get_or_build(path, build_fn, *args, **kwargs):
    """
    Load from serialized form or build an object, saving the built
    object.

    Remaining arguments are provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def build_vocab(corpus, synonyms, antonyms):
    """
    Build a vocabulary with word frequencies for an entire corpus.

    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    for line in synonyms:
        tokens = line.strip().split()
        vocab.update(tokens)


    for line in antonyms:
        tokens = line.strip().split()
        vocab.update(tokens)
    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}

def init_nym_dict(vocab, d):
    for k, v in vocab.items():
        d[v[0]] = []

    return d

def build_syncab(syns, vocab):
    logger.info("Building synonyms from list of synonyms")
    synonyms = init_nym_dict(vocab, dict())
    for l in syns:
        tokens = l.strip().split()
        # vocab = {word : (id, freq)}
        ids =  [ vocab[t][0] for t in tokens ]
        synonyms[ids[0]] = ids[1:]

    logger.info("Done building synonyms")    
    return synonyms

def build_antcab(ants, vocab):
    logger.info("Building antonyms from list of antonyms")
    antonyms = init_nym_dict(vocab, dict())
    for l in ants:
        tokens = l.strip().split()
        # vocab = {word : (id, freq)}
        ids =  [ vocab[t][0] for t in tokens ]
        antonyms[ids[0]] = ids[1:]

    logger.info("Done building antonyms")    
    return antonyms


@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.

    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form

        (i_main, i_context, cooccurrence)

    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).

    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]

# sim(w1, w2) = v_w1 dot v_w2 + b_w1
def sim_cost(w1, w2, bias):
    return w1.dot(w2) + bias


def sym_cost(word, synonyms, bias):
    cost = 0
    for s in synonyms:
        cost += log(expit(sim_cost(word, s, bias)))

    return cost

def ant_cost(word, antonyms, bias):
    cost = 0
    for s in antonyms:
        cost += log(expit(-(sim_cost(word, s, bias))))

    return cost

def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    `data` is a pre-fetched data / weights list where each element is of
    the form

        (v_main, v_context,
         b_main, b_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    as produced by the `train_glove` function. Each element in this
    tuple is an `ndarray` view into the data structure which contains
    it.

    See the `train_glove` function for information on the shapes of `W`,
    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
    should be initialized.

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.
    """

    global_cost = 0

    # We want to iterate over data randomly so as not to unintentionally
    # bias the word vector contents
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, v_synonyms, v_antonyms, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1


        # attempt to make new cost function according to antonym word embed paper - ABRJ
        #sym_c = sym_cost(v_main, v_synonyms, b_main[0])
        # ant_c = ant_cost(v_main, v_antonyms, b_main[0])
        # nym_cost = beta * (sym_c + a * ant_c)
        # cost_inner = cooccurrence * log(expit(sim_cost(v_main, v_context, b_main[0])))

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = sim_cost(v_main, v_context, b_main[0]) + b_context[0] - log(cooccurrence)  

        sym_c = sym_cost(v_main, v_synonyms, b_main[0])
        ant_c = ant_cost(v_main, v_antonyms, b_main[0])

        # Compute cost
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        beta = 100
        a = 3.2
        nym_cost = beta * (sym_c + a * ant_c)
        cost = cooccurrence * (cost_inner ** 2) + nym_cost

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        #
        # NB: `main_word` is only a view into `W` (not a copy), so our
        # modifications here will affect the global weight matrix;
        # likewise for context_word, biases, etc.

        weighted_nym_cost = weight * cost_inner + nym_cost

        grad_main = weight * cost_inner  * v_context
        grad_context = weight * cost_inner  * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2


    return global_cost




def train_glove(vocab, synonyms, antonyms, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form

        (word_i_id, word_j_id, x_ij)

    where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
    matrix defined by `build_cooccur` and the Pennington et al. (2014)
    paper itself.

    If `iter_callback` is not `None`, the provided function will be
    called after each iteration with the learned `W` matrix so far.

    Keyword arguments are passed on to the iteration step function
    `run_iter`.

    Returns the computed word vector matrix `W`.
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # It is up to the client to decide what to do with the resulting two
    # vectors. Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # Bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # Build a reusable list from the given cooccurrence generator,
    # pre-fetching all necessary data.
    #
    # NB: These are all views into the actual data matrices, so updates
    # to them will pass on to the real data structures
    #
    # (We even extract the single-element biases as slices so that we
    # can use them as views)
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             [W[s] for s in synonyms[i_main]],
             [W[a] for a in antonyms[i_main]],
             cooccurrence)
                for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W

def save_model(W, path):
    with open(path, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def main(arguments):
    corpus = arguments.corpus
    synonyms = arguments.synonyms
    antonyms = arguments.antonyms

    logger.info("Fetching vocab..")
    vocab = get_or_build(arguments.vocab_path, build_vocab, corpus, synonyms, antonyms)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Building list of synonyms and antonyms")
    synonyms = build_syncab(synonyms, vocab)
    antonyms = build_antcab(antonyms, vocab)
    logger.info("Loaded %i synonyms and %i antonyms", len(synonyms), len(antonyms))

    logger.info("Fetching cooccurrence list..")
    corpus.seek(0)
    cooccurrences = get_or_build(arguments.cooccur_path,
                                 build_cooccur, vocab, corpus,
                                 window_size=arguments.window_size,
                                 min_count=arguments.min_count)
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
                len(cooccurrences))

    if arguments.save_often:
        iter_callback = partial(save_model, path=arguments.vector_path)
    else:
        iter_callback = None

    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, synonyms, antonyms, cooccurrences,
                    iter_callback=iter_callback,
                    vector_size=arguments.vector_size,
                    iterations=arguments.iterations,
                    learning_rate=arguments.learning_rate)

    # TODO shave off bias values, do something with context vectors
    save_model(W, arguments.vector_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")
    main(parse_args())
