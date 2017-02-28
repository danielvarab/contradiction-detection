import argparse
import fnmatch
import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate


# returns a tuple: (embedding_name, list of predictions)
def readPredictionsFromPredictFile(path, prediction_file):
    prediction_files = {}
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, prediction_file):
            with open(os.path.join(root, filename), 'r') as f:
                predictions = f.readlines()
            embedding_name = os.path.basename(root)
            prediction_files[embedding_name] = predictions

    return prediction_files


def calculate_average(labels, predictions, index):
    average = 0.0
    number_of_embeddings = len(predictions)


    for key, value in predictions.iteritems():
        if (value[index] == labels[index]):
            average += 1

    if (average > 0):
        average = average / number_of_embeddings

    return average

def getPredictions(labels, predictions):
    result = []
    for index, label in enumerate(labels):
        if(label == predictions[index]):
            result.insert(index, 1)
        else:
            result.insert(index, 0)

    return result

def compute(sentA, sentB, labels, predictions):
    average = []
    sentA_l = []
    sentB_l = []

    for index, sentenceA in enumerate(sentA):
        avg = calculate_average(labels, predictions, index)
        sentA_l.append(len(sentenceA))
        sentB_l.append(len(sentB[index]))
        average.append(avg)


    df = pd.DataFrame()
    df['sentA'] = sentA
    df['sentA_l'] = sentA_l
    df['sentB'] = sentB
    df['sentB_l'] = sentB_l
    df['label'] = labels
    df['avr_pred'] = average

    for key, values in predictions.iteritems():
        df[key] = getPredictions(labels, values)

    # display options
    pd.set_option('display.width', 1000)
    pd.options.display.max_colwidth = 200
    return df


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

    predictions = readPredictionsFromPredictFile(args.directory, 'pred.txt')
    result = compute(sentenceA, sentenceB, labels, predictions)
    result.to_csv("output.txt", sep='\t')
    #print tabulate(result, headers='keys', tablefmt='psql')
    #print(result)

    # np.savetxt(args.outfile, confusion, delimiter=",", fmt="%1.0f")
