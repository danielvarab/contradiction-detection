import argparse
import sys
import os
from read_write import read_lexicon
from operator import itemgetter


def count_syns_ants(synonyms, antonyms, vocab):
	max_words = []
	min_words = []

	if vocab is None:
		print "vocab is none"
		vocab = set(synonyms.keys()).intersection(set(antonyms.keys()))
		print "resulting vocab of length: " + str(len(vocab))
	else:
		print "vocab from embedding included, with length: " + str(len(vocab))
		vocab = set(synonyms.keys()).intersection(set(antonyms.keys())).intersection(set(vocab))
		print "resulting vocab of length: " + str(len(vocab))

	for word in vocab:
		syn_length = 0
		ant_length = 0
		for s in synonyms[word]:
			if s in vocab:
				syn_length += 1
		for a in antonyms[word]:
			if s in vocab:
				ant_length += 1
		max_words.append((word,syn_length+ant_length))
		min_words.append((word,min(syn_length,ant_length)))


	return sorted(max_words, key=itemgetter(1), reverse=True)[1:10], sorted(min_words, key=itemgetter(1), reverse=True)[1:10]


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--s', required=True, help="synonym file")
	parser.add_argument('--a', required=True, help="antonym file")
	parser.add_argument('--e', required=False, help="optional word embedding for filtering")
	args = parser.parse_args(sys.argv[1:])

	synonyms = read_lexicon(args.s)
	antonyms = read_lexicon(args.a)

	vocab = None
	if args.e is not None:
		with open(args.e,"r") as f:
			vocab = [r.split()[0] for r in f]

	max_words, min_words = count_syns_ants(synonyms, antonyms, vocab)

	print "max words"
	print max_words
	print "min words"
	print min_words
