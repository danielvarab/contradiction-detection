import numpy as np
from ranking import *

""" antonym selection """
def antonym_selection(embedding, tasks, metric):
	results = []
	distance_func = None
	mean_vector = np.mean(np.array(embedding.values()), axis=0)
	for word, relations, gold in tasks:
		word_emb = embedding.get(word, mean_vector)

		index = np.inf
		if metric == "cosine":
			distance_func = lambda x: cosine_sim(embedding.get(x, mean_vector), word_emb)
			index = np.argmin(map(distance_func, relations))

		elif metric == "dot":
			distance_func = lambda x: np.dot(embedding.get(x, mean_vector), word_emb)
			index = np.argmin(map(distance_func, relations))

		else: raise ValueError("metric passed to antonym_selection is invalid {}".format(metric))

		if relations[index] == gold:
			results.append(1)
		else:
			results.append(0)

	return np.average(results)
