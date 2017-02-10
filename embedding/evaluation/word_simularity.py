import numpy as np
from ranking import cosine_sim, spearmans_rho, assign_ranks

import sys
import os

"""
	INPUT:
		embedding
		RETURNS: spearmanr correlation
"""
def word_similarity(embedding, X, y):
	scores = []
	# used to replace embeddings that are not contained in the vocabolary
	matrix = np.array(embedding.values(), dtype=np.float32)
	mean_vector = np.mean(matrix, axis=0)
	for w1, w2 in X:
		w_emb1, w_emb2 = embedding.get(w1, mean_vector), embedding.get(w2, mean_vector)
		score = cosine_sim(w_emb1, w_emb2)
		scores.append(score)

	return stats.spearmanr(scores, y).correlation

def evaluate__all_ws(embedding, task_dir):
	scores = {}
	for filename in os.listdir(task_dir):
		f = open(filename)
		X = []
		y = []
		for line in open(os.path.join(task_dir, filename),'r'):
			w1, w2, gold = line.strip().lower()
			X.append((w1,w2))
			y.append(gold)

		score = word_similarity(embedding, X, y)
		scores[filename] = score

	for score in scores:
		print(score, scores[score])


def eval_all_sim(word_vecs, word_sim_dir):
	results = {}
	for i, filename in enumerate(os.listdir(word_sim_dir)):
		manual_dict, auto_dict = ({}, {})
		not_found, total_size = (0, 0)
		for line in open(os.path.join(word_sim_dir, filename),'r'):
			line = line.strip().lower()
			word1, word2, val = line.split()
			if word1 in word_vecs and word2 in word_vecs:
				manual_dict[(word1, word2)] = float(val)
				auto_dict[(word1, word2)] = cosine_sim(word_vecs[word1], word_vecs[word2])
			else:
				not_found += 1
			total_size += 1
		score = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
		# print("> {0}: \t{1} (spearmans) - skipped {2}/{3}".format(filename, score, not_found, total_size))
		results[filename] = score
	return results
