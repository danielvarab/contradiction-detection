import os
import sys
import argparse
import numpy as np
from eval_sentiment import load_sentiment_data, test_embedding_on_task
from word_simularity import eval_all_sim
from synonym_selection import syn_ant_selection
from syntactic_relation import *
from read_write import *

def rel_path(path):
 return os.path.join(os.path.dirname(__file__), path)


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
		ws_path = rel_path('word_sim_tasks')
		eval_all_sim(embedding, ws_path)
	else:
		print(">> Skipped Word Similarity")

	if args.ss:
		print("\n")
		print('======================================')
		print('Synonym/Antonym Selection Evaluation')
		print('======================================')
		gre_path = rel_path("synonym_selection_tasks/testset950.txt")
		tasks = load_toefle(gre_path)
		precision = syn_ant_selection(embedding, tasks)
		print("> {0}: \t{1} (accuracy) - skipped {2}/{3}".format("testset950 (GRE)", precision, "TODO", "uses mean vector"))
	else:
		print(">> Skipped Synonym Selection (TOEFL)")

	# Sentiment Analysis (SA)
	if args.sa:
		print("\n")
		print('======================================')
		print('Sentiment Analysis Evaluation')
		print('======================================')
		sa_train_path = rel_path("sentiment_analysis/train.txt")
		sa_test_path = rel_path("sentiment_analysis/test.txt")
		SA_score = test_embedding_on_task(embedding, sa_train_path, sa_test_path)
		print("> {0}: \t{1} (accuracy) - skipped {2}/{3}".format("stanford-tree", SA_score, "TODO", "(uses mean vector)"))
		print("\n")

	else:
		print(">> Skipped Sentiment Analysis")
