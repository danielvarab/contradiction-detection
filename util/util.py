import numpy as np
def merge_main_context(W, merge_fun=lambda m, c: np.mean([m, c], axis=0),
                       normalize=True):
    """
    Merge the main-word and context-word vectors for a weight matrix
    using the provided merge function (which accepts a main-word and
    context-word vector and returns a merged version).

    By default, `merge_fun` returns the mean of the two vectors.
    """

    vocab_size = len(W) / 2
    for i, row in enumerate(W[:vocab_size]):
        merged = merge_fun(row, W[i + vocab_size])
        if normalize:
            merged /= np.linalg.norm(merged)
        W[i, :] = merged

    return W[:vocab_size]

def read_lines_in_file(path):
    with open(path) as f:
        return f.read().split("\n")