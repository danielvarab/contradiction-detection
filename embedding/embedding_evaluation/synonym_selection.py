import numpy as np
from ranking import *

""" synonym_selection (TOEFL) """
def syn_ant_selection(embedding, tasks):
	results = []
	distance_func = None
	mean_vector = np.mean(np.array(embedding.values()), axis=0)
	for word, relations, gold in tasks:
		placeholder = np.arange(300)
		word_emb = embedding.get(word, mean_vector)
		distance_func = lambda x: cosine_sim(embedding.get(x, mean_vector), word_emb)
		index = np.argmin(map(distance_func, relations))

		if relations[index] == gold:
			results.append(1)
		else:
			results.append(0)

	return np.average(results)
