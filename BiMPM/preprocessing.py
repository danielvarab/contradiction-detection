import numpy as np
import nltk
import json

from sklearn import preprocessing

tokenizer = nltk.tokenize.TreebankWordTokenizer()

UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO      = '**GO**'  # it's called "GO" but actually serves as a null alignment

DIMENSIONS = 300
def generate_char_encoder(chars):
    label_encoder = preprocessing.LabelEncoder()
    return label_encoder.fit(chars)

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
    char_vocab = set()
    max_sentence_length = 0
    max_word_length = 0
    with open(filename, 'rb') as f:
        if filename.endswith('.tsv') or filename.endswith('.txt'): raise ValueError("txt. no.")

        for line in f:
            line = line.decode('utf-8')
            if lowercase:
                line = line.lower()
            data = json.loads(line)
            if data['gold_label'] == '-': # ignore items without a gold label
                continue
            sentence1 = data["sentence1"]
            sentence2 = data["sentence2"]

            # update char_vocab
            char_vocab = char_vocab | set(sentence1) | set(sentence2)

            # determine max sentence length
            s1_tokens = tokenizer.tokenize(sentence1)
            s2_tokens = tokenizer.tokenize(sentence2)
            tmp = max(len(s1_tokens), len(s2_tokens))
            if tmp > max_sentence_length:
                max_sentence_length = tmp

            # determine max word length
            s1_word_lengths = [len(w) for w in s1_tokens]
            s2_word_lengths = [len(w) for w in s2_tokens]

            tmp2 = max(max(s1_word_lengths), max(s2_word_lengths))
            if tmp2 > max_word_length:
                max_word_length = tmp2

            t = (sentence1, sentence2, data['gold_label'])
            useful_data.append(t)
    char_vocab.remove(" ")
    return useful_data, max_sentence_length, max_word_length, list(char_vocab)

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

def sentence_to_char_vecs(sentence, encoder, max_sentence_length, max_word_length):
    words = []
    for word in tokenizer.tokenize(sentence):
        word_placeholder = np.zeros(max_word_length) # placeholder to contain the word in a moment
        char_encoded_word = encoder.transform(list(word)) + 1 # transform the word into labels, e.g: "what" => [1,2,3,4]. obs. +1 to avoid zeros
        word_length = char_encoded_word.shape[0] # length of the word
        word_placeholder[:word_length] = char_encoded_word # insert into placeholder vector
        words.append(word_placeholder)

    words = np.array(words, dtype=np.float32) # combine them
    sentence_placeholder = np.zeros((max_sentence_length, max_word_length)) # create placeholder for sentence
    word_count = words.shape[0] # get length count of words
    sentence_placeholder[:word_count] = words # insert words into placeholder
    return sentence_placeholder # returns.. you asshole


def label_to_vec(label): # one hotter
    label_dic = { "neutral":0, "entailment":1, "contradiction":2 }
    label = label_dic[label]
    vec = np.zeros(3)
    vec[label] = 1
    return vec

def npify(x):
    return np.array(x, dtype=np.float32)

def create_dataset(samples, embedding, encoder, max_sentence_length, max_word_length):
    a_sentences = []
    a_char_sentences = []
    b_sentences = []
    b_char_sentences = []
    labels = []
    for sample in samples:
        s1, s2, label = sample

        sentence_1_vecs = sentence_to_vecs(s1, embedding, max_sentence_length)
        sentence_2_vecs = sentence_to_vecs(s2, embedding, max_sentence_length)
        sentence_1_char_vecs = sentence_to_char_vecs(s1, encoder, max_sentence_length, max_word_length)
        sentence_2_char_vecs = sentence_to_char_vecs(s2, encoder, max_sentence_length, max_word_length)
        label_vec = label_to_vec(label)

        a_sentences.append(sentence_1_vecs)
        a_char_sentences.append(sentence_1_char_vecs)
        b_sentences.append(sentence_2_vecs)
        b_char_sentences.append(sentence_2_char_vecs)

        labels.append(label_vec)

    a_sentences = npify(a_sentences)
    b_sentences = npify(b_sentences)
    a_char_sentences = npify(a_char_sentences)
    b_char_sentences = npify(b_char_sentences)
    labels = npify(labels)

    return (a_sentences, a_char_sentences, b_sentences, b_char_sentences, labels)
