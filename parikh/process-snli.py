import os
import sys
import argparse
import numpy as np
import csv



def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_folder', help="location of folder with the snli files")
    parser.add_argument('--out_folder', help="location of the output folder")
    
    args = parser.parse_args(arguments)


    for split in ["train", "dev", "test"]:
        src_out = open(os.path.join(args.out_folder, "src-"+split+".txt"), "w")
        targ_out = open(os.path.join(args.out_folder, "targ-"+split+".txt"), "w")
        label_out = open(os.path.join(args.out_folder, "label-"+split+".txt"), "w")
        label_avr_out = open(os.path.join(args.out_folder, "label_average-"+split+".txt"), "w")

        label_set = set(["neutral", "entailment", "contradiction"])

        for line in open(os.path.join(args.data_folder, "snli_1.0_"+split+".txt"),"r"):
            d = line.split("\t")
            label = d[0].strip()
            premise = " ".join(d[1].replace("(", "").replace(")", "").strip().split())
            hypothesis = " ".join(d[2].replace("(", "").replace(")", "").strip().split())

            label1 = d[9].strip()
            label2 = d[10].strip()
            label3 = d[11].strip()
            label4 = d[12].strip()
            label5 = d[13].strip()
            tmp = label1 + "," + label2 + "," + label3 + "," + label4 + "," + label5
            list = tmp.split(",")
            labels_count = 0
            for s in list:
                if (s == label):
                    labels_count += 1

            if label in label_set:
                src_out.write(premise + "\n")
                targ_out.write(hypothesis + "\n")
                label_out.write(label + "\n")
                label_avr_out.write(str(labels_count) + "\n")

        src_out.close()
        targ_out.close()
        label_out.close()
        label_avr_out.close()



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
