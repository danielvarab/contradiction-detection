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
result = []
for line in snli:
    d = line.split("\t")
    gold_label = d[0].strip()
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
gold_labels = open('/home/contra/contradiction-detection/datasets/parikh-models/glove.840B.300d/label-test.txt', 'r').read().splitlines()
df = pd.DataFrame()
embeddings = {}
acc = 0
for dirpath, subdirs, files in os.walk(path):
    for file in files:
        if(file == "pred.txt"):
            path = os.path.join(dirpath, file)
            embedding_name = os.path.basename(os.path.dirname(path))
            labels = open(path, 'r').read().splitlines()
            predictions = [0] * len(gold_labels)
            for index, label in enumerate(labels):
                if label.strip() == gold_labels[index]:
                    predictions[index] = 1
                    acc += 1

            embeddings[embedding_name] = predictions
            acc = acc/len(labels)


df = pd.DataFrame.from_dict(embeddings,  dtype=int)
#print(df.columns)
#print(df.sample(2))


result_dataframe = df_result.join(df)


# Number of entailments
df_result[(df_result['gold_label'] == 'entailment')]

# Number of contradictions
df_result[(df_result['gold_label'] == 'contradiction')]

# Number of neutrals
df_result[(df_result['gold_label'] == 'neutral')]

# Number of non-gold_label eg. "-"
df_result[(df_result['gold_label'] == '-')]



list = ['glove.6B.300d', 'glove.840B.300d', 'glove.840B.300d.new_anto_rf_out.not-normalized', 'mce-0.4', 'glove.840B'
                                                                                                         '.300d+mce']
# Number of contradictions with antonyms relation
for embedding in list:
    l = len((result_dataframe[(result_dataframe['gold_label'] == 'contradiction') & (result_dataframe.astype(str)['ant_pairs'] != '[]') & (result_dataframe[embedding] == 1.0) ]))
    print("EMBEDDINGNAME " + embedding + " --> " + str(l))


# Number of contradictions with antonyms relation
len((df_result[(df_result['gold_label'] == 'contradiction') & (df_result.astype(str)['ant_pairs'] != '[]') ]))

# Number of entailments with antonyms relation
len((df_result[(df_result['gold_label'] == 'entailment') & (df_result.astype(str)['ant_pairs'] != '[]') ]))

# Number of neutrals with antonyms relation
len((df_result[(df_result['gold_label'] == 'neutral') & (df_result.astype(str)['ant_pairs'] != '[]') ]))

# Number of - with antonyms relation
len((df_result[(df_result['gold_label'] == '-') & (df_result.astype(str)['ant_pairs'] != '[]') ]))


# Number of contradictions with synonyms relation
len((df_result[(df_result['gold_label'] == 'contradiction') & (df_result.astype(str)['syn_pairs'] != '[]') ]))

# Number of entailments with synonyms relation
len((df_result[(df_result['gold_label'] == 'entailment') & (df_result.astype(str)['syn_pairs'] != '[]') ]))

# Number of neutrals with synonyms relation
len((df_result[(df_result['gold_label'] == 'neutral') & (df_result.astype(str)['syn_pairs'] != '[]') ]))

# Number of - with synonyms relation
len((df_result[(df_result['gold_label'] == '-') & (df_result.astype(str)['syn_pairs'] != '[]') ]))



# INTERSECTION OF SNLI VOCAB & LEXICON
snli_vocab = []
ant_vocab = []
syn_vocab = []

with open('/home/contra/contradiction-detection/parikh/glove.6B.300d.txt_flipFalse_2.0_50.0_anto_rf/entail.word.dict', 'r') as f:
    for line in f:
        snli_vocab.append(line.split()[0])

with open('/home/contra/contradiction-detection/datasets/ant_syn/synonym.txt', 'r') as f:
    for line in f:
        syn_vocab.append(line.split()[0])


with open('/home/contra/contradiction-detection/datasets/ant_syn/antonym.txt', 'r') as f:
    for line in f:
        ant_vocab.append(line.split()[0])


#print(len(set(snli_vocab) & set(ant_vocab)))

#print(len(set(snli_vocab) & set(syn_vocab)))



# GET SAMPLE SENTENCES
# get sentences labelled neutral that have some synonym pairs
test_sample = df_result[(df_result['gold_label'] == 'neutral') & (df_result.astype(str)['syn_pairs'] != '[]')][['sent1','sent2', 'syn_pairs']].sample(10); test_sample['sent1'].to_csv('sent1_sample.txt', index=False); test_sample['sent2'].to_csv('sent2_sample.txt', index=False); test_sample['syn_pairs'].to_csv('syn_pairs', index=False)

#  get sentences labelled neutral and does not have any synonym or antonym pairs
test_sample = df_result[(df_result['gold_label'] == 'neutral') & (df_result.astype(str)['syn_pairs'] == '[]') & (df_result.astype(str)['ant_pairs'] == '[]')][['sent1','sent2', 'syn_pairs']].sample(10); test_sample['sent1'].to_csv('sent1_sample.txt', index=False); test_sample['sent2'].to_csv('sent2_sample.txt', index=False); test_sample['syn_pairs'].to_csv('syn_pairs', index=False)


# GET INTERSECTIONS
# Getting all pairs only answered correctly by 'glove.6B.300d_new_anto_rf_out-FIXED'
test_intercetion = result_dataframe[(result_dataframe['glove.6B.300d']==0) & (result_dataframe['glove.6B.300d_new_anto_rf_out-FIXED']==1)]

# Getting all contradicting pairs, containing antonyms, only answered by 'glove.6B.300d_new_anto_rf_out-FIXED'
test_intercetion = result_dataframe[(result_dataframe['glove.6B.300d']==0) & (result_dataframe['glove.6B.300d_new_anto_rf_out-FIXED']==1) & (result_dataframe.astype(str)['ant_pairs'] != '[]') & (result_dataframe['gold_label'] == 'contradiction')];
len(test_intercetion)
test_intercetion[['sent1','sent2','ant_pairs']]

# More with print to csv
test_intercetion = result_dataframe[(result_dataframe['glove.840B.300d']==0) & (result_dataframe['glove.840B.300d.txt_new_anto_rf_out']==1)&(result_dataframe.astype(str)['ant_pairs'] != '[]') & (result_dataframe['gold_label'] == 'contradiction')];test_intercetion['sent2'].to_csv('sent2_sample.txt', index=False);test_intercetion['sent1'].to_csv('sent1_sample.txt', index=False); test_intercetion['ant_pairs'].to_csv('ant_pairs.txt', index=False); 










