import argparse
import fnmatch
import sys
import os
import numpy as np

def getPredictionsFromEmbeddings(path, filename):
    prediction_files = {}
    for root, subFolders, files in os.walk(path):
        if filename in files:
            with open(os.path.join(root, filename), 'r') as f:
                predictions = f.readlines()
            embedding_name = os.path.dirname(root)
            prediction_files[embedding_name] = predictions
    return prediction_files

def calculate_average(labels, predictions, index):
    result = {}
    result['embeddings'] = []
    average = 0
    number_of_embeddings = len(predictions)
    for key, value in predictions.iteritems():
        if(value[index] == labels[index]):
            average += 1
            result['embeddings'].append(key)
    average = average/number_of_embeddings
    result['average'] = average

    return result


def compute(sentA, sentB, labels, predictions):
    result = {}
    for index in enumerate(sentA):
        result['sentenceA'] = sentA[index]
        result['sentenceB'] = sentB[index]
        result['sentenceA_length'] = len(sentA[index])
        result['sentenceB_length'] = len(sentB[index])
        result['label'] = labels[index]
        calc = calculate_average(labels, predictions, index)
        result['average_prediction'] = calc['average']
        for key, value in predictions.iteritems():
            if(key in calc['embeddings']):
                result[key] = 1
            else:
                result[key] = 0

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--srctestfile', help="Path to sent1 test data.")
    parser.add_argument('--targettestfile', help="Path to sent2 test data.")
    parser.add_argument('--labeltestfile', help="Path to label test data.")
    parser.add_argument('--directory', help="Path to directories with prediction file.")

    args = parser.parse_args(sys.argv[1:])

    with open(args.srctestfile, 'r') as f:
        sentenceA = f.readlines()

    with open(args.targettestfile, 'r') as f:
        sentenceB = f.readlines()

    with open(args.labeltestfile, 'r') as f:
        labels = f.readlines()

    predictions = getPredictionsFromEmbeddings(args.directory, 'pred.txt')

    # np.savetxt(args.outfile, confusion, delimiter=",", fmt="%1.0f")
