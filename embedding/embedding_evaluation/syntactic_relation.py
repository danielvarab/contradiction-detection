import numpy as np
from ranking import *


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
	best_fit = None
	if distance_metric is "euclid":
		best_fit = (None, np.inf)
	if distance_metric is "cosine":
		best_fit = (None, 0)

	for word, embedding in embedding.iteritems():
		if distance_metric is "euclid":
			d = euclidean(vector, embedding)
			if d < best_fit[1]:
				best_fit = (word, d)

		if distance_metric is "cosine":
			d = cosine_sim(vector, embedding)
			if d > best_fit[1]:
				best_fit = (word, d)

	return best_fit[0]
