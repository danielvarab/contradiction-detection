import numpy as np 
import sys
import os
import argparse

from scipy.spatial import distance
import read_write 

def calculate_mean_distance(lexicon, words, distance_metric='cosine'):
	vocab = set(words.keys())
	distances = []

	for word in vocab:
		lexical_words_in_vocab = set(lexicon.get(word, [])).intersection(vocab)
		for i, lex in enumerate(lexical_words_in_vocab):
			if(distance_metric == 'cosine'):
				distances.append(distance.cosine(words[word],words[lex]))
			if(distance_metric == 'euclidean'):
				distances.append(distance.euclidean(words[word],words[lex]))			

	return float(np.mean(distances))

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

	syn_mean_dist = 0
	ant_mean_dist = 0

	if(args.d is not None):
		syn_mean_dist = calculate_mean_distance(synonyms, wordVecs, args.d)
		ant_mean_dist = calculate_mean_distance(antonyms, wordVecs, args.d)
	else:
		syn_mean_dist = calculate_mean_distance(synonyms,wordVecs)
		ant_mean_dist = calculate_mean_distance(antonyms,wordVecs)


	print('>> Distances in ' + args.e)
	print('>> Synonym mean distance: ' + str(syn_mean_dist))
	print('>> Antonym mean distance: ' + str(ant_mean_dist))








