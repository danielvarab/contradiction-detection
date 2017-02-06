import numpy as np
from ranking import *

""" synonym_selection (TOEFL) """
def synonym_selection(embedding, tasks, distance_metric="euclid"):
	results = []
	distance_func = None
	mean_vector = np.mean(np.array(embedding.values()), axis=0)
	for word, relations, gold in tasks:
		placeholder = np.arange(300)
		word_emb = embedding.get(word, mean_vector)
		if distance_metric is "euclid":
			distance_func = lambda x: euclidean(embedding.get(x, mean_vector), word_emb)
		if distance_metric is "cosine":
			distance_func = lambda x: cosine_sim(embedding.get(x, mean_vector), word_emb)
		index = np.argmax(map(distance_func, relations))
		if relations[index] == gold:
			results.append(1)
		else:
			results.append(0)

	return np.average(results)
