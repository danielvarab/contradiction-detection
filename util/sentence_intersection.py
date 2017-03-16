import pandas as pd
import os
# INPUT
snli = open('/home/contra/contradiction-detection/datasets/snli_1.0/snli_1.0_test.txt', 'r')



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
gold_labels = []
result = []
for line in snli:
    d = line.split("\t")
    gold_label = d[0].strip()
    gold_labels.append(gold_label)
    sent1 = " ".join(d[1].replace("(", "").replace(")", "").strip().split())
    sent2 = " ".join(d[2].replace("(", "").replace(")", "").strip().split())

    label1 = d[9].strip()
    label2 = d[10].strip()
    label3 = d[11].strip()
    label4 = d[12].strip()
    label5 = d[13].strip()
    tmp = label1 + "," + label2 + "," + label3 + "," + label4 + "," + label5
    list = tmp.split(",")
    neutrals = 0
    contradictions = 0
    entailments = 0
    if (list[1] and list[2] and list[3] and list[4]):
        for l in list:
            if(l == "neutral"): neutrals += 1
            if l == "contradiction": contradictions += 1
            if(l == "entailment"): entailments += 1
    else:
        if(gold_label == "neutral"): neutrals = 5
        elif(gold_label == "contradiction"): contradictions = 5
        elif(gold_label == "entailment"): entailments = 5

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

    result.append((sent1, sent2, gold_label, neutrals, contradictions, entailments, syn_pairs, ant_pairs))
    columns = ["sent1", "sent2", "gold_label", "neutrals", "contradictions", "entailments", "syn_pairs", "ant_pairs"]
    df_result = pd.DataFrame(result, columns=columns)



#Filter result on antonyms and contradictions
#antonyms_result = []
#contradictions = []

#for el in result:
#    if(el[4]):
#        antonyms_result.append(el)
#        if(el[2] == "contradiction"):
#            contradictions.append(el)


path = "/home/contra/contradiction-detection/datasets/parikh-models/"

df = pd.DataFrame()
embeddings = {}
for dirpath, subdirs, files in os.walk(path):
    for file in files:
        if(file == "pred.txt"):
            path = os.path.join(dirpath, file)
            embedding_name = os.path.basename(os.path.dirname(path))
            labels = open(path, 'r').read().splitlines()
            #print(len(gold_labels))
            predictions = [0] * len(labels)
            for index, label in enumerate(labels):
                #print("LABEL: " + label + " GOLD_LABEL: " + gold_labels[index])
                #predictions.insert(index, 0)
                if label == gold_labels[index]:
                    predictions[index] = 1

            embeddings[embedding_name] = predictions


df = pd.DataFrame.from_dict(embeddings,  dtype=int)
#print(df.columns)
#print(df.sample(2))


result_dataframe = df_result.join(df)
