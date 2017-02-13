import argparse
import sys
import os
import numpy as np


def calculate_confusion(labels, predict):
	assert len(labels)==len(predict)
	idx_dict = {'contradiction':0,'entailment':1,'neutral':2}
	value = np.asarray([(0,0,0),(0,0,0),(0,0,0)], dtype=int)
	for i in range(len(labels)):
		l = labels[i].lower().strip()
		p = predict[i].lower().strip()
		value[idx_dict[l]][idx_dict[p]] += 1

	return value

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--labels', default=None, help="file with list of original labels")
	parser.add_argument('--predict', default=None, help="file with list of original labels")
	parser.add_argument('--outfile', help="Outfile name")

	args = parser.parse_args(sys.argv[1:])

	with open(args.labels, 'r') as l:
		labels = l.readlines()

	with open(args.predict, 'r') as p:
		predict = p.readlines()

	confusion = calculate_confusion(labels, predict)
	np.savetxt(args.outfile, confusion, delimiter=",", fmt="%1.0f")
