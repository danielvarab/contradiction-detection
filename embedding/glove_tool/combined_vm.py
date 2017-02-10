import cPickle as pickle
import msgpack
import numpy as np
import sys
import re
from argparse import ArgumentParser


def read_vocab(path):
    with open(path, 'rb') as file_obj:
        return msgpack.load(file_obj, use_list=False, encoding='utf-8')

def read_model(path):
    with open(path, 'rb') as file_obj:
        return pickle.load(file_obj)

def save_file(vocab, model, output_name):
    # vocab in this form: Word -> (freq, id)
    output_file = open(output_name, 'w+')
    for word, tuple in vocab.iteritems():
        vector_id = tuple[1]
        output_file.write(word+' ')
        for val in model[vector_id]:
            output_file.write(np.array_str(val)+' ')
        output_file.write('\n')



