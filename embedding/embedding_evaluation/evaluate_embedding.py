import sys
import argparse
import numpy as np
from random import shuffle
from scipy import stats
from scipy.spatial import distance
from similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from analogy import fetch_semeval_2012_2
from sklearn.metrics import pairwise_distances
from eval_sentiment import load_sentiment_data, test_embedding_on_task


'''Load GRE or TOEFLE task'''
def load_toefle(path) :
	task = []
	with open(path, 'rb') as ant_file:
		for l in ant_file:
			a = l.split(':')
			task.append((a[0].strip(), a[1].strip().split(' ') ,a[3].strip()))

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

def syntactic_relations(embedding, pairs):
	count = 0 # debuggin purposes
	total = len(pairs)*len(pairs)

	results = []
	for (a,b) in pairs:
		for (c,d) in pairs:
			# debuggin purposes
			count += 1
			print (count, total)

			if a is c and b is d:
				print("skipped as pairs are equal")
				continue

			q = embedding[a] - embedding[b] + embedding[c]
			nearest_word = nearest_neighbour(embedding, q)
			# debuggin purposes

			print ("d:{}".format(d), "guessed:{}".format(nearest_word))

			if nearest_word == d:
				results.append(1)
			else:
				results.append(0)

	return np.average(results)


"""
	RETURNS: nearest word in the embedding with respect to input argument 'vector'
"""
def nearest_neighbour(embedding, vector, distance_metric="cosine"):
	best_fit = (None, np.inf)
	for word, embedding in embedding.iteritems():
		d = 0
		if distance_metric is "euclid":
			d = distance.euclidean(vector, embedding)
		if distance_metric is "cosine":
			d = distance.cosine(vector, embedding)

		if d < best_fit[1]:
			best_fit = (word, d)

	return best_fit[0]

def toefle(embedding, tasks, distance_metric="euclid"):
	results = []
	distance_func = None
	mean_vector = np.mean(np.array(embedding.values()), axis=0)
	for word, relations, gold in tasks:
		placeholder = np.arange(300)
		word_emb = embedding.get(word, mean_vector)
		if distance_metric is "euclid":
			distance_func = lambda x: distance.euclidean(embedding.get(x, mean_vector), word_emb)
		if distance_metric is "cosine":
			distance_func = lambda x: distance.cosine(embedding.get(x, mean_vector), word_emb)
		index = np.argmax(map(distance_func, relations))
		if relations[index] == gold:
			results.append(1)
		else:
			results.append(0)

	return np.average(results)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', help="embedding file")
	parser.add_argument('--v', help="vocabolary file", default=None)
	parser.add_argument('--t', help="toefl file", default=None)
	parser.add_argument('--syn_rel', help="Syntactic Relatedness file", default=None) # this can be empty as we fetch it
	parser.add_argument('--sa_train', help="Sentiment Analysis train data", default=None)
	parser.add_argument('--sa_test', help="Sentiment Analysis test data", default=None)

	args = parser.parse_args(sys.argv[1:])

	print("> Loading embedding into memory")
	embedding = {}
	if args.v is not None:
		embedding = load_embedding_from_two_files(args.e, args.v)
	else:
		embedding = load_embedding(args.e)

	print("> Done loading embedding")

	print("> Starting evaluations of embedding...")

	print("> Starting Word Simularity Evaluations")
	tasks = {
		"MEN": fetch_MEN(),
		"RG-65": fetch_RG65(),
		"WS-353": fetch_WS353()
	}
	for task, data in tasks.iteritems():
		score = evaluate_similarity(embedding, data.X, data.y)
		print(task, score)

	# syntactic relatedness
	if args.syn_rel is not None:
		print("> Starting Syntatic Relatedness evaluation")
		semeval = fetch_semeval_2012_2() # categories
		for category, pairs in semeval.X.iteritems():
			shuffle(pairs) # in-place.... alright python
			precision = syntactic_relations(embedding, pairs[:2])
			print(category, precision)
	else:
		print(">> Skipped syn-rel. no file (doesn't actually need file)")

	if args.t is not None:
		print("> Starting Synonym Selection (TOEFL) evaluation")
		tasks = load_toefle(args.t)
		precision = toefle(embedding, tasks)
		print("toefl", precision)
	else:
		print(">> Skipped Synonym Selection (TOEFL)")

	# Sentiment Analysis (SA)
	if args.sa_train is not None and args.sa_test is not None:
		print("> Starting Sentiment Analysis Evaluation")
		SA_score = test_embedding_on_task(embedding, args.sa_train, args.sa_test)
		print("Sentiment Analysis", SA_score)
	else:
		print(">> Skipped Sentiment Analysis")
