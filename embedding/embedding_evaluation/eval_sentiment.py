import numpy as np
import math
import sys
import codecs
from nltk import Tree
from sklearn import linear_model

"""
    INPUT:
        file: name of file that contains the embeddings.
              the formatting of the file is dictionary-like <word> <embedding>.
              (seperated by whitespace. first entry denotes the key)
"""
def load_embedding(emb_file):
    with open(emb_file, "r") as f:
        dic = {}
        rows = f.readlines()
        for row in rows:
            attributes = row.split()
            dic[attributes[0]] = np.array(attributes[1:]).astype(float)
        return dic

def load_sentiment_data(path):
	sentiments = []
	sentences = []
	count = 0
	with codecs.open(path, 'rb', 'utf8') as task_file:
		for l in task_file:
			tree = Tree.fromstring(l)
		  	flat = tree.flatten()
		  	#Transform from 5 to 2 label problem
		  	
		  	if (int(flat.label())<2): #adjust negatives
		  		sentiments.append(0)
				sentences.append(flat.leaves())
		  	elif (int(flat.label())>2): #adjust positives
		  		sentiments.append(1)
				sentences.append(flat.leaves())
			
	return sentiments, sentences

def average_sentences_vector(sentences, embed):
	vectors = np.zeros((len(sentences), 300))
	skip_count = 0
	skips = []
	for i, s in enumerate(sentences):
		length = len(s)
		vec = np.zeros(300)
		for w in s:
			w = w.lower()
			if w not in embed:
				length-1
				skip_count += 1
				skips.append(w)
			else:
				vec = vec + embed[w]
		vectors[i] = (vec/length)
	#print('skipped words: ' + str(skip_count))
	#print(skips)
	return vectors

def train_sentiment_classifyer(vectors, sentiment, cs):
	logreg = linear_model.LogisticRegressionCV(Cs=cs)
	logreg.fit(vectors,sentiment)
	return logreg

def test_embedding_on_task(embed):
	#Load train and test data
	train_sentiments, train_sentences = load_sentiment_data("../../datasets/trees/train.txt")
	test_sentiments, test_sentences = load_sentiment_data("../../datasets/trees/test.txt")
	#Average word vectors in sentences in training and test data
	train_vectors = average_sentences_vector(train_sentences,embed)
	test_vectors = average_sentences_vector(test_sentences,embed)
	#Train classifier on training data, and score on test data
	logreg = train_sentiment_classifyer(train_vectors,train_sentiments, [1.0,1e2,1e3,1e4,1e5])
	return logreg.score(test_vectors,test_sentiments)

if __name__ == "__main__":
	#Load embeddings
	embed = load_embedding("../../datasets/glove_data/glove.6B/glove.6B.300d.txt")

	print (test_embedding_on_task(embed))










