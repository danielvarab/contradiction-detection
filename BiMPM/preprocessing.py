import numpy as np
import nltk
import json

tokenizer = nltk.tokenize.TreebankWordTokenizer()
UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO      = '**GO**'  # it's called "GO" but actually serves as a null alignment

DIMENSIONS = 300

def generate_vector(shape): return np.random.uniform(-0.1, 0.1, shape)

def load_embeddings(emb_file, normalize=False, to_lower=False):
    word_vectors = {}
    if emb_file.endswith('.gz'): f = gzip.open(emb_file, 'r')
    else: f = open(emb_file, 'r')

    word_vectors[UNKNOWN] = generate_vector(DIMENSIONS)
    word_vectors[PADDING] = generate_vector(DIMENSIONS)
    word_vectors[GO] = generate_vector(DIMENSIONS)

    for row in f:
        # row = row.decode('utf-8')
        row = row.strip()
        row = row.split()
        if row == '':
            continue

        vector = np.array(row[1:], dtype=np.float32)

        if to_lower:
            word = word.lower()
        if normalize:
            vector /= math.sqrt((vector**2).sum() + 1e-6)
        word_vectors[row[0]] = vector

    return word_vectors


def read_corpus(filename, lowercase=True):
    useful_data = []
    max_sentence_length = 0
    with open(filename, 'rb') as f:
        if filename.endswith('.tsv') or filename.endswith('.txt'): raise ValueError()

        for line in f:
            line = line.decode('utf-8')
            if lowercase:
                line = line.lower()
            data = json.loads(line)
            if data['gold_label'] == '-': # ignore items without a gold label
                continue
            sentence1 = data["sentence1"]
            sentence2 = data["sentence2"]
            s1_tokens = tokenizer.tokenize(sentence1)
            s2_tokens = tokenizer.tokenize(sentence2)
            tmp = max(len(s1_tokens), len(s2_tokens))
            if tmp > max_sentence_length:
                max_sentence_length = tmp
            t = (sentence1, sentence2, data['gold_label'])
            useful_data.append(t)

    return useful_data, max_sentence_length

def sentence_to_vecs(sentence, word_dict, sentence_length):
    shape = (sentence_length, DIMENSIONS)
    placeholder = np.full(shape, word_dict[PADDING], dtype=np.float32)

    vecs = []
    tokens = tokenizer.tokenize(sentence)
    for index, word in enumerate(tokens):
        word_vec = word_dict.get(word, None)
        if word_vec is None:
            word_vec = word_dict[UNKNOWN] # OOV
        placeholder[index] = word_vec

    return placeholder

def label_to_vec(label): # one hotter
    label_dic = { "neutral":0, "entailment":1, "contradiction":2 }
    label = label_dic[label]
    vec = np.zeros(3)
    vec[label] = 1
    return vec

def npify(x):
    return np.array(x, dtype=np.float32)

def create_dataset(corpus_path, embedding_path):
    # word_dictionary :: dictionary :: word => vector
    embedding = load_embeddings(embedding_path)

    # tuples :: tuple :: (sentence1, sentence2, label)
    tuples, max_sentence_length = read_corpus(corpus_path)

    a_sentences = []
    b_sentences = []
    labels = []
    for entry in tuples:
        s1, s2, label = entry

        sentence_1_vecs = sentence_to_vecs(s1, embedding, max_sentence_length)
        sentence_2_vecs = sentence_to_vecs(s2, embedding, max_sentence_length)
        label_vec = label_to_vec(label)

        a_sentences.append(sentence_1_vecs)
        b_sentences.append(sentence_2_vecs)

        labels.append(label_vec)

    a_sentences = npify(a_sentences)
    b_sentences = npify(b_sentences)
    labels = npify(labels)

    return (a_sentences, b_sentences, labels, max_sentence_length)
