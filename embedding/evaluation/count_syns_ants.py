import argparse
import sys
import os
from read_write import read_lexicon


def count_syns_ants(synonyms, antonyms, vocab):
	max_word = ""
	min_word = ""
	max_count = 0
	min_count = 0
	if vocab is None:
		print "vocab is none"
		vocab = set(synonyms.keys()).intersection(set(antonyms.keys()))
		print "resulting vocab of length: " + str(len(vocab))
	else:
		print "vocab from embedding included, with length: " + str(len(vocab))
		vocab = set(synonyms.keys()).intersection(set(antonyms.keys())).intersection(set(vocab))
		new_syns_keys = vocab.intersection(synonyms)
		synonyms = {k:synonyms[k] for k in new_syns_keys}
		new_ants_keys = vocab.intersection(antonyms)
		antonyms = {k:antonyms[k] for k in new_ants_keys}
		print "resulting vocab of length: " + str(len(vocab))

	for word in vocab:
		length = len(synonyms[word])+len(antonyms[word])
		min_length = min(len(synonyms[word]),len(antonyms[word]))
		if (length > max_count):
			max_count = length
			max_word = word
		if (min_length > min_count):
			min_count = min_length
			min_word = word

	return max_word, max_count, min_word, min_count 


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

	max_word, max_count, min_word, min_count  = count_syns_ants(synonyms, antonyms, vocab)


	print max_word
	print max_count
	print len(synonyms[max_word])
	print len(antonyms[max_word])
	print min_word
	print min_count
	print len(synonyms[min_word])
	print len(antonyms[min_word])
