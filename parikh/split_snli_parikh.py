import pandas as pd
import numpy as np
import argparse
import sys
import os
import csv


def split_data(arguments):
    # train
    file_path = arguments.trainfile
    print(file_path)
    write_file(pd.read_csv(file_path, sep="\t"), 550000, arguments.output, "train")

    # development
    file_path = arguments.devfile
    print(file_path)
    write_file(pd.read_csv(file_path, sep="\t"), 10000, arguments.output, "dev")

    # validation
    file_path = arguments.testfile
    print(file_path)
    remove_quotes = lambda x:x.replace('"', '')
    write_file(pd.read_csv(file_path, sep="\t", converters={'param': remove_quotes, 'value': remove_quotes}), 10000, arguments.output, "val")



def write_file(df, numberOfSentences, output, type):
    headers = ["sentence1", "sentence2", "gold_label"]
    if(len(df.index) <= numberOfSentences):
        sliced = df[[headers[0], headers[1], headers[2]]]
    else:
        sliced = df[[headers[0], headers[1], headers[2]]].sample(n=numberOfSentences)
    
    # sort based on length of sentence1
    sliced["sentence1_length"] = (sliced.sentence1.str.len())
    sliced = sliced.sort_values(["sentence1_length"], ascending=True)
    
    # remove all non-gold_label values
    sliced = sliced[sliced[headers[2]] != "-"]
    #remove quotes
    #sliced["sentence1"] = sliced.sentence1.str.replace('"', '')
    #sliced["sentence2"] = sliced['sentence2'].map(lambda x: x.replace("\"", ""))
    
    
    # append NULL
    sliced['sentence1'] = 'NULL ' + sliced['sentence1'].astype(str)
    sliced['sentence2'] = 'NULL ' + sliced['sentence2'].astype(str)
    #print(sliced[["sentence1","sentence1_length"]])
    
    #sliced["sentence1"] = sliced.sentence1.str.replace('"', '')
    #sliced["sentence2"] = sliced.sentence2.str.replace('"', '')

    for header in headers:
        res = sliced[header]
        outputName = output + type + "-" + header + "-" + str(numberOfSentences) + "-SNLI.txt"
        res.to_csv(outputName, index=False)
        print("Created file with " + str(numberOfSentences) + " " + header + " pairs")



def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', help="Number of sentences",
                        type=int, default=5000)
    parser.add_argument('--devfile', help="Path to SNLI development set.")
    parser.add_argument('--trainfile', help="Path to SNLI training set.")
    parser.add_argument('--testfile', help="Path to SNLI validation set.")
    parser.add_argument('--output', help="Path to outputfolder.",
                        default="output/")
    args = parser.parse_args(arguments)

    directory = args.output
    if not os.path.exists(directory):
        os.makedirs(directory)

    split_data(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

