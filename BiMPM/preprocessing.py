import numpy as np
import nltk
import json

from sklearn import preprocessing

tokenizer = nltk.tokenize.TreebankWordTokenizer()

UNKNOWN = '**UNK**'
PADDING = '**PAD**'
GO      = '**GO**'  # it's called "GO" but actually serves as a null alignment

DIMENSIONS = 300

SENTENCES_1_FILE = "data/sentences1.txt"
SENTENCES_2_FILE = "data/sentences2.txt"
LABEL_FILE = "data/labels.txt"

def generate_char_encoder(chars):
    label_encoder = preprocessing.LabelEncoder()
    return label_encoder.fit(chars)

def generate_vector(shape): return np.random.uniform(-0.1, 0.1, shape)

def npify(x): return np.array(x, dtype=np.float32)

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

            # determine max sentence and word length
            max_sentence_length = longest_sentence(sentence1, sentence2, max_sentence_length)
            max_word_length = longest_word(sentence1, sentence2, max_word_length)

            t = (sentence1, sentence2, data['gold_label'])
            useful_data.append(t)
    char_vocab.remove(" ")
    return useful_data, max_sentence_length, max_word_length, list(char_vocab)

def preprocess_corpus(filename, lowercase=True):
    """
        This function takes a file, and produces a variaty of this.
        1. 3 output files, which each contain sentence 1, sentence 2, and the label.
        The returned output of this function is the stats of the dataset
    """
    sample_count = 0
    char_vocab = set()
    max_sentence_length = 0
    max_word_length = 0
    with open(filename, 'rb') as f,\
         open(SENTENCES_1_FILE, "w+") as s1_file,\
         open(SENTENCES_2_FILE, "w+") as s2_file,\
         open(LABEL_FILE, "w+") as label_file:
        for line in f:
            line = line.decode('utf-8')
            if lowercase: line = line.lower()

            data = json.loads(line)
            if data['gold_label'] == '-': # ignore items without a gold label
                continue

            sentence1 = data["sentence1"]
            sentence2 = data["sentence2"]
            label = data['gold_label']

            s1_file.write(sentence1 + "\n")
            s2_file.write(sentence2 + "\n")
            label_file.write(label + "\n")

            # increment count
            sample_count += 1

            # update char_vocab
            char_vocab = char_vocab | set(sentence1) | set(sentence2)

            # determine max sentence and word length
            max_sentence_length = longest_sentence(sentence1, sentence2, max_sentence_length)
            max_word_length = longest_word(sentence1, sentence2, max_word_length)

        char_vocab.remove(" ") # don't encode whitespace.. would be no words
        return {
            "sentence_length": max_sentence_length,
            "word_length": max_word_length,
            "sample_count": sample_count,
            "char_vocab": list(char_vocab)
        }

def longest_sentence(s1, s2, current_max):
    s1_tokens = tokenizer.tokenize(s1)
    s2_tokens = tokenizer.tokenize(s2)
    return max(len(s1_tokens), len(s2_tokens), current_max)

def longest_word(s1, s2, current_max):
    s1_word_lengths = [len(w) for w in tokenizer.tokenize(s1)]
    s2_word_lengths = [len(w) for w in tokenizer.tokenize(s2)]
    return max(max(s1_word_lengths), max(s2_word_lengths), current_max)

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

def _to_word_vecs(embedding, max_sentence_length):
    def curried(sentence):
        return sentence_to_vecs(sentence, embedding, max_sentence_length)
    return curried

def _to_char_vecs(encoder, max_sentence_length, max_word_length):
    def curried(sentence):
        return sentence_to_char_vecs(sentence, encoder, max_sentence_length, max_word_length)
    return curried

def label_to_vec(label): # one hotter
    label_dic = { "neutral":0, "entailment":1, "contradiction":2 }
    label = label_dic[label.rstrip()]
    vec = np.zeros(3)
    vec[label] = 1
    return vec

def create_dataset(samples, embedding, encoder, max_sentence_length, max_word_length):
    a_sentences = []
    a_char_sentences = []
    b_sentences = []
    b_char_sentences = []
    labels = []

    word_veccer = _to_word_vecs(embedding, max_sentence_length)
    char_veccer = _to_char_vecs(encoder, max_sentence_length, max_word_length)
    for sample in samples:
        s1, s2, label = sample

        sentence_1_vecs = word_veccer(s1)
        sentence_2_vecs = word_veccer(s2)
        sentence_1_char_vecs = char_veccer(s1)
        sentence_2_char_vecs = char_veccer(s2)
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

def create_dataset_generator(embedding, encoder, max_sentence_length, max_word_length, batch_size=32):
    while 1:
        sentences1_f = open(SENTENCES_1_FILE, "r")
        sentences2_f = open(SENTENCES_2_FILE, "r")
        labels_f = open(LABEL_FILE, "r")

        a_sentences = []
        a_char_sentences = []
        b_sentences = []
        b_char_sentences = []
        labels = []

        word_veccer = _to_word_vecs(embedding, max_sentence_length)
        char_veccer = _to_char_vecs(encoder, max_sentence_length, max_word_length)

        for index, (s1,s2,label) in enumerate(zip(sentences1_f, sentences2_f, labels_f)):
            sentence_1_vecs = word_veccer(s1)
            sentence_2_vecs = word_veccer(s2)
            sentence_1_char_vecs = char_veccer(s1)
            sentence_2_char_vecs = char_veccer(s2)
            label_vec = label_to_vec(label)

            a_sentences.append(sentence_1_vecs)
            a_char_sentences.append(sentence_1_char_vecs)
            b_sentences.append(sentence_2_vecs)
            b_char_sentences.append(sentence_2_char_vecs)

            labels.append(label_vec)

            if len(labels) == batch_size:
                yield({
                    "word_sentence_A": npify(a_sentences),
                    "char_sentence_A": npify(a_char_sentences),
                    "word_sentence_B": npify(b_sentences),
                    "char_sentence_B": npify(b_char_sentences)
                }, {"output": npify(labels)})

                a_sentences = []
                a_char_sentences = []
                b_sentences = []
                b_char_sentences = []
                labels = []
        yield({
            "word_sentence_A": npify(a_sentences),
            "char_sentence_A": npify(a_char_sentences),
            "word_sentence_B": npify(b_sentences),
            "char_sentence_B": npify(b_char_sentences)
        }, {"output": npify(labels)})

        sentences1_f.close()
        sentences2_f.close()
        labels_f.close()


# model.fit_generator(generate_arrays_from_file('/my_file.txt'), samples_per_epoch=10000, nb_epoch=10)
