import numpy as np
from autograd import grad

synonyms = {}
antonyms = {}

def sim(vec1, vec2): return np.dot(vec1,vec2)

def objective(embedding):
    alpha = 1; beta = 1; gamma = 1
    cost = 0

    for word in words:
        s_nabors = synonyms[word]
        a_nabors = antonyms[word]
        cost += left(word, alpha) + midle(word, beta, s_nabors) + right(word, gamma, a_nabors)

    return cost

# origin embedding
def left(word, alpha): # alpha was one ?
    return alpha * sim(new_word_embs[word], word_embs[word])

# synonyms
def middle(word, beta, nabors): # beta was the amount of nabors?
    value = 0
    for nabor in nabors:
        qi = new_word_embs[name]
        qj = new_word_embs[nabor]
        result = sim(qi, qj)
        value = value + (beta * result)

    return value

# antonyms
def right(word, gamma): # what is gamma?
    value = 0
    for nabor in lex[word]:
        qi = new_word_embs[name]
        qk = new_word_embs[nabor]
        result = sim(qi, qj)
        value = value + (gamma * -result)

    return value


""" Simularity function  """
def sim(vec1, vec2):
    return np.dot(vec1.A1, vec2.A1)

''' Retrofit word vectors to a lexicon '''
def retrofit_v2(wordVecs, synonyms, antonyms, numIters):
    lookup = {}
    vectors = []
    count = 0
    for word, v in wordVecs.iteritems():
        vectors.append(v)
        lookup[word] = count
        count += 1

    matrix = np.matrix(vectors)
    newMatrix = matrix.copy()

    newWordVecs = deepcopy(wordVecs)
    embedding_words = set(newWordVecs.keys())
    synonym_words = set(synonyms.keys())
    antonym_words = set(antonyms.keys())

    loopVocab = embedding_words.intersection(synonym_words.union(antonym_words))

    def objective(embedding):
        alpha = 1; beta = 1; gamma = 1 # currently fixed biases

        for word in loopVocab:
            # lookup for index of word-embedding
            word_index = lookup[word]

            # part of equation that computes the cost for maintaining the existing position
            origin_cost = alpha * sim(embedding[word_index], matrix[word_index])

            # part of equation computes the cost for synonyms
            synonym_cost = 0
            for synonym in synonyms[word]:
                if synonym in lookup:
                    syn_index = lookup[synonym]
                    synonym_cost = synonym_cost + (beta * sim(embedding[word_index], embedding[syn_index]))

            # part of equation that computes the cost for antonyms
            antonym_cost = 0
            for antonym in antonyms[word]:
                if antonym in lookup:
                    ant_index = lookup[antonym]
                    antonym_cost = antonym_cost + (gamma * -sim(embedding[word_index], embedding[ant_index]))

        return origin_cost + synonym_cost + antonym_cost

    update_func = grad(objective)

    for it in range(numIters):
        newMatrix = newMatrix + update_func(newMatrix)

    return newWordVecs

if __name__ == "__main__":
    derivative = grad(objective)
    inputs = {
        "a": [0.52, 1.12,  0.77],
        "b": [0.52, 0.06, -1.30],
        "c": [0.74, -2.49, 1.39]
    }

    print(derivative(inputs))
