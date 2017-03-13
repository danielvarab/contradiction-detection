# coding: utf-8


# INPUT
sent1_file = open('/home/contra/contradiction-detection/datasets/parikh-models/GC_ACL2012/src-train.txt','r')
sent2_file = open('/home/contra/contradiction-detection/datasets/parikh-models/GC_ACL2012/targ-train.txt','r')
labels = open('/home/contra/contradiction-detection/datasets/parikh-models/GC_ACL2012/label-train.txt','r')

sent1_lines = sent1_file.read().splitlines()
sent2_lines = sent2_file.read().splitlines()
labels_lines = labels.read().splitlines()
snli = list(zip(sent1_lines, sent2_lines, labels_lines))



# Generate antonyms/synonyms dictionaries based on words from snli
antonyms = open('/home/contra/contradiction-detection/embedding/retrofitting/lexicons/antonym.txt', 'r')
synonyms = open('/home/contra/contradiction-detection/embedding/retrofitting/lexicons/synonym.txt', 'r')
antonyms_dict = {}
for l in antonyms:
    words = l.lower().strip().split()
    antonyms_dict[words[0]] = words[1:]

synonyms_dict = {}
for l in synonyms:
    words = l.lower().strip().split()
    synonyms_dict[words[0]] = words[1:]


# Generate result data structure
result = []
for sent1, sent2, label in snli:
    sent2_set = set(sent2.split())
    syn_pairs = []
    ant_pairs = []
    for word in sent1.split():
        word_syn_set = set(synonyms_dict.get(word, []))
        word_ant_set = set(antonyms_dict.get(word, []))
        if (len(word_syn_set) > 0):
            synonyms = (word_syn_set & sent2_set)
            for s in synonyms:
                syn_pairs.append((word, s))
        if (len(word_ant_set) > 0):
            antonyms = (word_ant_set & sent2_set)
            for a in antonyms:
                ant_pairs.append((word, a))

    result.append((sent1, sent2, label, syn_pairs, ant_pairs))


# Filter result on antonyms and contradictions
antonyms_result = []
contradictions = []

for el in result:
    if(el[4]):
        antonyms_result.append(el)
        if(el[2] == "contradiction"):
            contradictions.append(el)


