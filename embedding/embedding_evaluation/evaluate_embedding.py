from __future__ import print_function

import os
import sys
import argparse
import numpy as np

from os import listdir
from os.path import isfile, join

from eval_sentiment import load_sentiment_data, test_embedding_on_task
from word_simularity import eval_all_sim
from synonym_selection import antonym_selection
from syntactic_relation import *
from read_write import *

def rel_path(path):
 return os.path.join(os.path.dirname(__file__), path)

# thanks buddy
# http://stackoverflow.com/a/14981125/2194536
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def tab_print(list):
	print("{:>50} {:>16} {:>16} {:>16} {:>16} {:>16}".format(*list))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', default=None, help="embedding file")
	parser.add_argument('--d', default=None, help="embedding directory, containing embedding files")
	parser.add_argument('--ws', action="store_true", default=False)
	parser.add_argument('--ss', action="store_true", default=False)
	parser.add_argument('--sa', action="store_true", default=False)

	args = parser.parse_args(sys.argv[1:])

	if args.e is None and args.d is None:
		print("need some sort of embedding through --e or --d")
		sys.exit()

	embedding_files = [args.e]
	if args.d is not None:
		embedding_files = [join(args.d, f) for f in listdir(args.d) if isfile(join(args.d, f))]

	tab_print([ "Embedding", "MEN", "RG-65", "WS-353", "GRE", "SA" ])

	for e_file in embedding_files:
		eprint("> Loading embedding into memory from {}".format(e_file))
		# embedding = read_word_vectors(args.e)
		embedding = load_embedding(emb_file=e_file, normalize=True)
		eprint("> Done loading embedding from {}".format(e_file))

		results = {}

		eprint("> Starting evaluations of embedding...")
		if args.ws:
			ws_path = rel_path('word_sim_tasks')
			rs = eval_all_sim(embedding, ws_path)
			results.update(rs)
		else:
			print(">> Skipped Word Similarity")

		if args.ss:
			gre_path = rel_path("synonym_selection_tasks/testset950.txt")
			tasks = load_toefle(gre_path)
			precision = antonym_selection(embedding, tasks)
			results["GRE"] = precision
		else:
			eprint(">> Skipped Antonym Selection")

		# Sentiment Analysis (SA)
		if args.sa:
			sa_train_path = rel_path("sentiment_analysis/train.txt")
			sa_test_path = rel_path("sentiment_analysis/test.txt")
			SA_score = test_embedding_on_task(embedding, sa_train_path, sa_test_path)
			results["SA"] = SA_score

		else:
			eprint(">> Skipped Sentiment Analysis")

		skipped = "skipped"
		MEN = results.get("EN-MEN-TR-3k.txt", skipped)
		RG65 = results.get("EN-RG-65.txt", skipped)
		WS353 = results.get("EN-WS-353-ALL.txt", skipped)
		GRE = results.get("GRE", skipped)
		SA = results.get("SA", skipped)


		tab_print([ e_file.split("/")[-1], MEN, RG65, WS353, GRE, SA])

		eprint(">> Done evaluating {}\n".format(e_file))
