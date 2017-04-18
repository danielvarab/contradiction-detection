import numpy as np 
import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.spatial import distance
from scipy.stats import describe, normaltest, mannwhitneyu
from random import sample
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

	if(distances is not []):
		mean = describe(distances).mean
	else:
		mean = None
	norm = normaltest(distances)

	return mean, norm.pvalue, distances

def calculate_lex_significance(lexicon1, lexicon2):
	return mannwhitneyu(lexicon1, lexicon2).pvalue

def plot_distance_distribution(lexicon1, lexicon2, control, label1, label2, conLabel, title, distance):
	sns.distplot(lexicon1, label=label1)
	sns.distplot(lexicon2, label=label2)
	sns.distplot(control, label=conLabel)
	plt.title(str(title))
	plt.legend()
	plt.xlabel(distance)
	#plt.show()
<<<<<<< 94ed0a342abae3b5e782cf3e40eb3781133a1cfa
	plt.savefig("figures/ant_syn_distribution_" + title + "_" + distance + ".png", format='png')
=======
	plt.savefig("figures/ant_syn_distribution_" + title + ".png", format='png')
>>>>>>> minor stuff...

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

	control = sample(wordVecs.values(),int(math.sqrt(len(ants)))*2)
	control_dist = []
	for v in control[0:len(control)/2]:
		for u in control[len(control)/2:]:
			if(args.d is not 'cosine'):
				control_dist.append(distance.cosine(v,u))
			else:
				control_dist.append(distance.euclidean(v,u))

	significance = calculate_lex_significance(syns, ants)
	print("Calculated significance...")

	plot_distance_distribution(syns, ants, control_dist, "synonyms", "antonyms", "control", os.path.splitext(os.path.basename(args.e))[0], args.d)

	print('>> Distances in ' + args.e)
	print('>> Synonym mean distance: ' + str(syn_mean_dist))
	print('>> Antonym mean distance: ' + str(ant_mean_dist))
	print('>> Synonym norm p-value: ' + str(syn_norm))
	print('>> Antonym norm p-value: ' + str(ant_norm))
	print('>> Distance sets significance: ' + str(significance))








