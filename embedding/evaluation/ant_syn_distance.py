import numpy as np 
import sys
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import distance
from scipy.stats import describe, normaltest, mannwhitneyu
import read_write 

def calculate_lex_distance(lexicon, words, distance_metric='cosine'):
	vocab = set(words.keys())
	distances = []

	for word in vocab:
		lexical_words_in_vocab = set(lexicon.get(word, [])).intersection(vocab)
		for i, lex in enumerate(lexical_words_in_vocab):
			if(distance_metric == 'cosine'):
				distances.append(distance.cosine(words[word],words[lex]))
			if(distance_metric == 'euclidean'):
				distances.append(distance.euclidean(words[word],words[lex]))

	desc = describe(distances)
	norm = normaltest(distances)

	return desc.mean, norm.pvalue, distances

def calculate_lex_significance(lexicon1, lexicon2):
	return mannwhitneyu(lexicon1, lexicon2).pvalue

def plot_distance_distribution(lexicon1, lexicon2, label1, label2, title):
	sns.distplot(lexicon1, label=label1)
	sns.distplot(lexicon2, label=label2)
	plt.title(str(title))
	plt.legend()
	plt.show()

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', required=True, help="embedding file")
	parser.add_argument('--s', required=True, help="synonym lexicon file")
	parser.add_argument('--a', required=True, help="antonym lexicon file")
	parser.add_argument('--d', required=False, help="Distance metric. Choose between euclidean and cosine")
	args = parser.parse_args(sys.argv[1:])

	name = args.e
	wordVecs = read_write.read_word_vectors(args.e)
	synonyms = read_write.read_lexicon(args.s)
	antonyms = read_write.read_lexicon(args.a)

	if(args.d is not None):
		syn_mean_dist, syn_norm, syns = calculate_lex_distance(synonyms, wordVecs, args.d)
		ant_mean_dist, ant_norm, ants = calculate_lex_distance(antonyms, wordVecs, args.d)
	else:
		syn_mean_dist, syn_norm, syns = calculate_lex_distance(synonyms,wordVecs)
		ant_mean_dist, ant_norm, ants = calculate_lex_distance(antonyms,wordVecs)
	print("Calculated distances...")

	significance = calculate_lex_significance(syns, ants)
	print("Calculated significance...")

	plot_distance_distribution(syns, ants, "synonyms", "antonyms", os.path.basename(str(args.e)))

	print('>> Distances in ' + args.e)
	print('>> Synonym mean distance: ' + str(syn_mean_dist))
	print('>> Antonym mean distance: ' + str(ant_mean_dist))
	print('>> Synonym norm p-value: ' + str(syn_norm))
	print('>> Antonym norm p-value: ' + str(ant_norm))
	print('>> Distance sets significance: ' + str(significance))








