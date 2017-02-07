import os
import sys
import argparse
import numpy as np
from eval_sentiment import load_sentiment_data, test_embedding_on_task
from word_simularity import eval_all_sim
from synonym_selection import synonym_selection
from syntactic_relation import *
from read_write import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', help="embedding file")
	parser.add_argument('--ws', action="store_true", default=False)
	parser.add_argument('--ss', action="store_true", default=False)
	parser.add_argument('--sa', action="store_true", default=False)

	args = parser.parse_args(sys.argv[1:])

	print("> Loading embedding into memory")
	# embedding = read_word_vectors(args.e)
	embedding = load_embedding(args.e, normalize=True)
	print("> Done loading embedding")

	print("> Starting evaluations of embedding...")

	if args.ws:
		print("\n")
		print('======================================')
		print('Word Simularity Evaluations')
		print('======================================')
		eval_all_sim(embedding, "./word_sim_tasks")
	else:
		print(">> Skipped Word Similarity")

	if args.ss:
		print("\n")
		print('======================================')
		print('Synonym Selection Evaluation')
		print('======================================')
		tasks = load_toefle("./synonym_selection_tasks/testset950.txt")
		precision = synonym_selection(embedding, tasks)
		print("> {0}: \t{1} (accuracy) - skipped {2}/{3}".format("testset950", precision, "TODO", "uses mean vector"))
	else:
		print(">> Skipped Synonym Selection (TOEFL)")

	# Sentiment Analysis (SA)
	if args.sa:
		print("\n")
		print('======================================')
		print('Sentiment Analysis Evaluation')
		print('======================================')
		SA_score = test_embedding_on_task(embedding, "./sentiment_analysis/train.txt", "./sentiment_analysis/test.txt")
		print("> {0}: \t{1} (accuracy) - skipped {2}/{3}".format("stanford-tree", SA_score, "TODO", "(uses mean vector)"))
		print("\n")

	else:
		print(">> Skipped Sentiment Analysis")
