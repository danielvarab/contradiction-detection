import sys
import argparse
import numpy as np

def load_embedding(emb_file):
	print("reading embedding from {0}".format(emb_file))
	word_vectors = {}
	f = open(emb_file)

	for row in f:
		row = row.rstrip().split()
		word = row[0].rstrip()
		vector = np.array(row[1:]).astype(np.float32)
		word = word.lower()
		word_vectors[word] = vector

	return word_vectors

''' Write word vectors to file '''
def print_word_vecs(wordVectors, outFileName):
	sys.stderr.write('\nWriting down the vectors in '+outFileName+'\n')
	outFile = open(outFileName, 'w')
	for word, values in wordVectors.iteritems():
		outFile.write(word+' ')
		for val in wordVectors[word]:
			outFile.write('%.4f' %(val)+' ')
		outFile.write('\n')
	outFile.close()

parser = argparse.ArgumentParser()
parser.add_argument('--emb1', required=True, help="large embedding file")
parser.add_argument('--emb2', required=True, help="small embedding file")
parser.add_argument('--out', required=True, help="out embedding file")

args = parser.parse_args(sys.argv[1:])

emb2 = load_embedding(args.emb2)
emb1 = load_embedding(args.emb1)
new_emb = {}

small_mean_vector = np.mean(np.array(emb2.values(), dtype=np.float32), axis=0)

for word, vector in emb1.iteritems():
	# in hope that this will reduce the dictionaries memory use.
	# dno bout python though...
	vector2 = emb2.pop(word, small_mean_vector)

	new_emb[word] = np.concatenate((vector, vector2))

print_word_vecs(new_emb, args.out)
