import sys
import argparse
import numpy as np
from scipy import stats
from scipy.spatial import distance
from similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW


'''Load GRE or TOEFLE task'''
def load_toefle(path) :
	task = []
	with open(path, 'rb') as ant_file:
		for l in ant_file:
			a = l.split(':')
			task.append((a[0].strip(),a[1].strip().split(' '),a[3].strip()))

	return task

"""
    INPUT (file names):
        name_file: vocabolary file. only single wordVectors
        vector_file: vector file. indexes relate to them of the name_file
    RETURNS:
        dictionary from string to vector
"""
def load_embedding_from_two_files(name_file, vector_file):
    with open(name_file, "r") as n_file, open(vector_file) as v_file:
        names = n_file.readlines()
        vectors = v_file.readlines()

        dic = { value.rstrip():np.array(vectors[index]).astype(float) for index, value in enumerate(names) }

        return dic

"""
    INPUT:
        file: name of file that contains the embeddings.
              the formatting of the file is dictionary-like <word> <embedding>.
              (seperated by whitespace. first entry denotes the key)
"""
def load_embedding(emb_file):
    with open(emb_file, "r") as f:
        dic = {}
        rows = f.readlines()
        for row in rows:
            attributes = row.split()
            dic[attributes[0]] = np.array(attributes[1:]).astype(float)
        return dic

"""
    INPUT:
        embedding
    RETURNS: spearmanr correlation
"""
def evaluate_similarity(embedding, X, y):
    scores = []
    # used to replace embeddings that are not contained in the vocabolary
    matrix = np.array(embedding.values()).astype(float)
    mean_vector = np.mean(matrix, axis=0)
    for w1, w2 in X:
        w_emb1, w_emb2 = embedding.get(w1, mean_vector), embedding.get(w2, mean_vector)
        scores.append(np.dot(w_emb1, w_emb2))

    return stats.spearmanr(scores, y).correlation


def syntactic_relations(embedding, test_tuples):
    results = []
    for abcd in test_tuples:
        q = abcd[0] - abcd[1] + abcd[2]
        word = closest_neighbour(embedding, q)

        if word == abcd[1]:
            results.append(1)
        else:
            results.append(0)

    return np.average(results)

def closest_neighbour(embedding, vector):
    best_fit = ("", 999999999999)
    for word, embedding in embedding.iteritems():
        dist = numpy.linalg.norm(vector-embedding)
        if distance < best_fit[1]:
            best_fit = (word, dist)

    return best_fit[0]

def toefle(word_embedding, task):
    euclid_correct = 0; cosine_correct = 0; skip_count = 0
    for t in task:
        word, antonyms, gold = t

        if word not in word_embedding:
            skip_count += 1
            continue

        word_emb = word_embedding[word]
        options = np.array(antonyms)
        euclidian = np.argmax(map(lambda option: distance.euclidean(word_embedding[option], word_emb), options))
		# cosine = np.argmax(map(lambda x: distance.cosine(word_embedding[x],source),options))
        if (antonyms[euclidian] == gold):
			euclid_correct += 1

        print(antonyms[euclidian], gold)
		# if (antonyms[cosine] == t[2]):
		# 	cosine_correct += 1

	return (euclid_correct,cosine_correct, skip_count, len(task))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--e', help="embedding file")
#     parser.add_argument('--v', help="vocabolary file", default=None)
#     parser.add_argument('--t', help="toefl file", default=None)
#
#     args = parser.parse_args(sys.argv[1:])
#
#     print("> Loading embedding into memory")
#     embedding = {}
#     if args.v is not None:
#         embedding = load_embedding_from_two_files(args.e, args.v)
#     else:
#         embedding = load_embedding(args.e)
#
#     print("> Done loading embedding")
#
#     tasks = {
#         # "MEN": fetch_MEN(),
#         # "RG-65": fetch_RG65(),
#         # "WS-353": fetch_WS353(),
#     }
#
#     print("> Starting evaluations of embedding")
#     for task, data in tasks.iteritems():
#         score = evaluate_similarity(embedding, data.X, data.y)
#
#         print(task, score)
#
#
#     if args.t is not None:
#         tasks = load_toefle(args.t)
#
#         euclidian_correct, consine_correct, skip_count, total_count = toefle(embedding, tasks)
#
#         print(euclidian_correct, total_count, skip_count)
