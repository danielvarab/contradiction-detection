import numpy as np
import math
import sys
import codecs
from nltk import Tree

def load_task(path):
	task = []
	count = 0
	with codecs.open(path, 'rb', 'utf8') as task_file:
		for l in task_file:
			if (count==0):
				count=1
				s = codecs.decode(l, 'utf8')
				print(type(s))
			tree = Tree.fromstring(l)
		  	#sentiment = str(d)[1] #sentiment root is index 1 in the string
		  	#sentence = str(d.to_lines()[0])
		  	#print(sentiment,sentence)
		  	flat = tree.flatten()
			task.append((int(flat.label()),flat.leaves()))
	return task

#def eval_embed_on_task(word_embedding, task):

task = load_task("../datasets/trees/dev.txt")

print(task[0])