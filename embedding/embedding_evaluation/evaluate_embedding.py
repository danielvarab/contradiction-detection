import sys
import argparse
import numpy as np
from scipy import stats
from similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW


"""
    INPUT (file names):
        name_file: vocabolary file. only single wordVectors
        vector_file: vector file. indexes relate to them of the name_file
    RETURNS:
        dictionary from string to vector
"""
def load_embedding_from_two_files(name_file, vector_file):
    with open(name_file, "r") as n_file, open(vector_file) as v_file:
        names = n_file.readlines()
        vectors = v_file.readlines()

        dic = { value.rstrip():np.array(vectors[index]).astype(float) for index, value in enumerate(names) }

        return dic

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

"""
    INPUT:
        embedding
    RETURNS: spearmanr correlation
"""
def evaluate_similarity(embedding, X, y):
    scores = []
    # used to replace embeddings that are not contained in the vocabolary
    matrix = np.array(embedding.values()).astype(float)
    mean_vector = np.mean(matrix, axis=0)
    for w1, w2 in X:
        w_emb1, w_emb2 = embedding.get(w1, mean_vector), embedding.get(w2, mean_vector)
        scores.append(np.dot(w_emb1, w_emb2))

    return stats.spearmanr(scores, y).correlation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--e', help="embedding file")
    parser.add_argument('--v', help="vocabolary file", default=None)

    args = parser.parse_args(sys.argv[1:])

    print("> Loading embedding into memory")
    embedding = {}
    if args.v is not None:
        embedding = load_embedding_from_two_files(args.e, args.v)
    else:
        embedding = load_embedding(args.e)

    print("> Done loading embedding")

    tasks = {
        "MEN": fetch_MEN(),
        "RG-65": fetch_RG65(),
        "WS-353": fetch_WS353(),
    }

    print("> Starting evaluations of embedding")
    for task, data in tasks.iteritems():
        score = evaluate_similarity(embedding, data.X, data.y)

        print(task, score)
