import sys
import numpy as np
import argparse
from evaluation import read_write
from retrofitting import retrofit

def random_embedding_from_existing(embedding):
	rand_embedding = {}
	sample_key, sample_val = embedding.popitem()
	length = len(sample_val)
	rand_embedding[sample_key] = np.random.rand(length)

	for key in embedding:
		rand_embedding[key] = np.random.rand(length)

	return rand_embedding


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--e', help="embedding file")
	parser.add_argument('--o', default=None, help="embedding file")


	args = parser.parse_args(sys.argv[1:])

	print("Loading embedding into memory...")
	embedding = read_write.load_embedding(args.e)

	print("Generating random embeddings based on loaded one...")
	rand_embedding = random_embedding_from_existing(embedding)

	if args.o is None:
		file_name = args.e.rsplit('.', 1)[0] + '.jpg'
		file_name = "{}{}".format(file_name, ".RANDOM.txt")
	else:
		file_name = args.o

	retrofit.print_word_vecs(rand_embedding, file_name)



