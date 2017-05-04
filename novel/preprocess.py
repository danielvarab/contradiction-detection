from os.path import basename
import os
import numpy as np

def get_embedding_matrix(fn, vocab_size, vocab_dim, tokenizer):
    filebase_name = ".".join(basename(fn).split(".")[:-1])
    EMB_STORE = '{0}.weights'.format(filebase_name)
    if os.path.exists(EMB_STORE + '.npy'):
        return np.load(EMB_STORE + '.npy')

    embeddings_index = {}
    f = open(fn)
    for line in f:
      line = line.rstrip()
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()

    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, vocab_dim))
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      else:
        print('Missing from embedding: {}'.format(word))

    np.save(EMB_STORE, embedding_matrix)
    return embedding_matrix
