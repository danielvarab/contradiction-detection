import numpy as np
import math
import sys
from scipy.spatial import distance

''' Read all the word vectors and normalize them '''
def read_word_vecs(filename):
	wordVectors = {}
 	if filename.endswith('.gz'): fileObject = gzip.open(filename, 'r')
 	else: fileObject = open(filename, 'r')

 	for line in fileObject:
 		print(len(wordVectors))
 		line = line.strip().lower()
    	word = line.split()[0]
    	wordVectors[word] = np.zeros(len(line.split())-1, dtype=float)
    	for index, vecVal in enumerate(line.split()[1:]):
      		wordVectors[word][index] = float(vecVal)
    	#''' normalize weight vector '''
    	#wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)

  	sys.stderr.write("Vectors read from: "+filename+" \n")
	return wordVectors


'''Load GRE or TOEFLE task'''
def load_task(path) :
	task = []
	with open(path, 'rb') as ant_file:
		for l in ant_file:
			a = l.split(':')
			task.append((a[0].strip(),a[1].strip().split(' '),a[3].strip()))

	return task


def eval_embed_on_task(word_embedding, task) :
	euclid_correct = 0
	cosine_correct = 0
	skip_count = 0
	if ('but' in word_embedding):
		print('great!')

	for t in task:
		if (t[0] not in word_embedding):
			skip_count += 1
			continue
		source = word_embedding[t[0]]
		options = np.array(t[1])
		euclidian = np.argmax(map(lambda x: distance.euclidian(word_embedding[x],source),options))
		cosine = np.argmax(map(lambda x: distance.cosine(word_embedding[x],source),options))
		if (t[1][euclidian] == t[2]):
			euclid_correct += 1
		if (t[1][cosine] == t[2]):
			cosine_correct += 1

	return (euclid_correct,cosine_correct, skip_count, len(task))


ant_task = load_task('../datasets/GRE/antonym/testset950.txt')
embed = read_word_vecs('../datasets/glove_data/glove.6B/glove.6B.50d.txt')

print(len(embed))

for key in embed:
	print (key)

#result = eval_embed_on_task(embed, ant_task)
#print(result)
