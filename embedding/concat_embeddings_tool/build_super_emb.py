import sys
import argparse
import numpy as np

def load_embedding(emb_file):
	word_vectors = {}
	f = open(emb_file)

	for row in f:
		row = row.split()
		word = row[0].rstrip()
		vector = np.array(row[1:], dtype=np.float32)
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
parser.add_argument('--emb1', required=True, help="first embedding file")
parser.add_argument('--emb2', required=True, help="second embedding file")
parser.add_argument('--out', required=True, help="out embedding file")

args = parser.parse_args(sys.argv[1:])

emb1 = load_embedding(args.emb1)
emb2 = load_embedding(args.emb2)
new_emb = {}

words = set(emb1.keys()).intersection(emb2.keys())

for word in words:
	vector1 = emb1.pop(word) # in hope that this will reduce the dictionaries memory use.
	vector2 = emb2.pop(word) # dno bout python though...
	new_emb[word] = np.concatenate((vector1, vector2))

print_word_vecs(new_emb, args.out)
