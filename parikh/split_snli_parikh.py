import pandas as pd
import numpy as np
import argparse
import sys
import os


def split_data(arguments):
    # train
    file_path = arguments.trainfile
    print(file_path)
    write_file(pd.read_csv(file_path, sep="\t"), arguments, "train")

    # development
    file_path = arguments.devfile
    print(file_path)
    write_file(pd.read_csv(file_path, sep="\t"), arguments, "dev")

    # validation
    file_path = arguments.testfile
    print(file_path)
    write_file(pd.read_csv(file_path, sep="\t"), arguments, "val")


def write_file(df, arguments, type):
    headers = ["sentence1", "sentence2", "gold_label"]
    if(len(df.index) <= arguments.n):
        sliced = df[[headers[0], headers[1], headers[2]]]
    else:
        sliced = df[[headers[0], headers[1], headers[2]]].sample(n=arguments.n)
    for header in headers:
        res = sliced[header]
        output = arguments.output + type + "-" + header + "-" + str(arguments.n) + "-SNLI.txt"
        res.to_csv(output, index=False)
        print("Created file with " + str(arguments.n) + " " + header + " pairs")



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

