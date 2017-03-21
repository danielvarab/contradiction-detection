from __future__ import print_function

import os
import sys
import argparse
import numpy as np

from os import listdir
from os.path import isfile, join

from eval_sentiment import load_sentiment_data, test_embedding_on_task
from word_simularity import eval_all_sim
from antonym_selection import antonym_selection
from syntactic_relation import *
from read_write import *
from ant_syn_distance import calculate_lex_distance, calculate_lex_significance


def rel_path(path):
    return os.path.join(os.path.dirname(__file__), path)


# thanks buddy
# http://stackoverflow.com/a/14981125/2194536
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def tab_print(list):
    s = "{:>50}"
    for i in range(len(list)-1):
        s = s + " {:>16}"
    print(s.format(*list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', default=None, help="embedding file")
    parser.add_argument('--d', default=None, help="embedding directory, containing embedding files")
    parser.add_argument('--tolower', action="store_true", default=False, help="parse embedding by tolowering")
    parser.add_argument('--ws', action="store_true", default=False)
    parser.add_argument('--ss', action="store_true", default=False)
    parser.add_argument('--sa', action="store_true", default=False)
    parser.add_argument('--dc', action="store_true", default=False,
                        help="calculate mean cosine distance for antonyms and synonyms")
    parser.add_argument('--de', action="store_true", default=False,
                        help="calculate mean euclidean distance for antonyms and synonyms")

    args = parser.parse_args(sys.argv[1:])

    if args.e is None and args.d is None:
        print("need some sort of embedding through --e or --d")
        sys.exit()

    embedding_files = [args.e]
    if args.d is not None:
        embedding_files = [join(args.d, f) for f in listdir(args.d) if isfile(join(args.d, f)) and f.endswith(".txt")]

    header = ["Embedding"]
    if(args.ws):
        header = header + ["MEN", "RG-65", "WS-353", "SIMLEX"]
    if(args.ss):
        header = header + ["GRE(cos)", "GRE(dot)"]
    if(args.sa):
        header = header + ["SA"]
    if(args.dc):
        header = header + ["DCsyn", "DCant", "DCsynnorm", "DCantnorm", "DCsign"]
    if(args.de):
        header = header + ["DEsyn", "DEant", "DEsynnorm", "DEantnorm", "DEsign"]

    tab_print(header)

    for e_file in embedding_files:
        eprint("> Loading embedding into memory from {}".format(e_file))
        # embedding = read_word_vectors(args.e)
        embedding = load_embedding(emb_file=e_file, normalize=True, to_lower=args.tolower)
        eprint("> Done loading embedding from {}".format(e_file))

        results = {}

        eprint("> Starting evaluations of embedding...")
        if args.ws:
            ws_path = rel_path('word_sim_tasks')
            rs = eval_all_sim(embedding, ws_path)
            results.update(rs)
        else:
            eprint(">> Skipped Word Similarity")

        if args.ss:
            gre_path = rel_path("synonym_selection_tasks/testset950.txt")
            tasks = load_toefle(gre_path)
            precision_cos = antonym_selection(embedding, tasks, metric="cosine")
            precision_dot = antonym_selection(embedding, tasks, metric="dot")
            results["GREc"] = precision_cos
            results["GREd"] = precision_dot
        else:
            eprint(">> Skipped Antonym Selection")

        # Sentiment Analysis (SA)
        if args.sa:
            sa_train_path = rel_path("sentiment_analysis/train.txt")
            sa_test_path = rel_path("sentiment_analysis/test.txt")
            SA_score = test_embedding_on_task(embedding, sa_train_path, sa_test_path)
            results["SA"] = SA_score

        else:
            eprint(">> Skipped Sentiment Analysis")

        if args.dc:
            syn_path = rel_path("../retrofitting/lexicons/synonym.txt")
            ant_path = rel_path("../retrofitting/lexicons/antonym.txt")
            synonyms = read_lexicon(syn_path)
            antonyms = read_lexicon(ant_path)
            syn_mean_dist, syn_norm, syn_dists = calculate_lex_distance(synonyms, embedding, 'cosine')
            ant_mean_dist, ant_norm, ant_dists = calculate_lex_distance(antonyms, embedding, 'cosine')
            significance = calculate_lex_significance(syn_dists, ant_dists)
            results["DCsyn"] = syn_mean_dist
            results["DCant"] = ant_mean_dist
            results["DCsynnorm"] = syn_norm
            results["DCantnorm"] = ant_norm
            results["DCsign"] = significance

        else:
            eprint(">> Skipped Mean Cosine Distance")

        if args.de:
            syn_path = rel_path("../retrofitting/lexicons/synonym.txt")
            ant_path = rel_path("../retrofitting/lexicons/antonym.txt")
            synonyms = read_lexicon(syn_path)
            antonyms = read_lexicon(ant_path)
            syn_mean_dist, syn_norm, syn_dists = calculate_lex_distance(synonyms, embedding, 'euclidean')
            ant_mean_dist, ant_norm, ant_dists = calculate_lex_distance(antonyms, embedding, 'euclidean')
            significance = calculate_lex_significance(syn_dists, ant_dists)
            results["DEsyn"] = syn_mean_dist
            results["DEant"] = ant_mean_dist
            results["DEsynnorm"] = syn_norm
            results["DEantnorm"] = ant_norm
            results["DEsign"] = significance

        else:
            eprint(">> Skipped Mean Euclidean Distance")

        skipped = "skipped"
        MEN = results.get("EN-MEN-TR-3k.txt", skipped)
        RG65 = results.get("EN-RG-65.txt", skipped)
        WS353 = results.get("EN-WS-353-ALL.txt", skipped)
        SIMLEX = results.get("EN-SIMLEX-999.txt", skipped)
        GREc = results.get("GREc", skipped)
        GREd = results.get("GREd", skipped)
        SA = results.get("SA", skipped)
        DCsyn = results.get("DCsyn", skipped)
        DCant = results.get("DCant", skipped)
        DCsynnorm = results.get("DCsynnorm", skipped)
        DCantnorm = results.get("DCantnorm", skipped)
        DCsign = results.get("DCsign", skipped)
        DEsyn = results.get("DEsyn", skipped)
        DEant = results.get("DEant", skipped)
        DEsynnorm = results.get("DEsynnorm", skipped)
        DEantnorm = results.get("DEantnorm", skipped)
        DEsign = results.get("DEsign", skipped)

        print_list = [e_file.split("/")[-1]]
        if(args.ws):
            print_list = print_list + [MEN, RG65, WS353, SIMLEX]
        if(args.ss):
            print_list = print_list + [GREc, GREd]
        if(args.sa):
            print_list = print_list + [SA]
        if(args.dc):
            print_list = print_list + [DCsyn, DCant, DCsynnorm, DCantnorm, DCsign]
        if(args.de):
            print_list = print_list + [DEsyn, DEant, DEsynnorm, DEantnorm, DEsign]

        eprint(">> Writing results...")
        tab_print(print_list)

        eprint(">> Done evaluating {}\n".format(e_file))
