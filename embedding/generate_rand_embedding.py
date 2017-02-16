import sys
import numpy as np
import argparse
import evaluation




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', default=None, help="embedding file")

	args = parser.parse_args(sys.argv[1:])

	embedding = evaluation.read_write.read_word_vectors(args.e)

