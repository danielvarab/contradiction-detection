import pandas as pd
import numpy as np
from sklearn import linear_model

delete_cost = 1
insertion_cost = 1
substitution_cost = 1

def delete(value): return value + delete_cost
def insert(value): return value + insertion_cost

def substitute(equal, value):
    if equal:
        return value
    else:
        return value+substitution_cost

def edit_distance(s1, s2):
    s1 = s1.split(" ")
    s2 = s2.split(" ")

    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0] = i
    for j in range(n): tbl[0,j] = j
    for i in range(1, m):
        for j in range(1, n):
            d_cost = delete(tbl[i, j-1])
            i_cost = insert(tbl[i-1, j])
            s_cost = substitute(s1[i-1] == s2[j-1], tbl[i-1, j-1])

            tbl[i,j] = min(d_cost, i_cost, s_cost)

    # print(tbl)
    return tbl[i,j]

file_path = "/home/danielvarab/school/contradiction-detection/sick_dataset/SICK_train.txt"
df = pd.read_csv(file_path, sep="\t")

equal_distribution = []
label_groups = df.groupby("entailment_judgment")
for x in label_groups.groups:
    group = label_groups.get_group(x)
    equal_distribution.append(group[:600])

df1 = pd.concat(equal_distribution, ignore_index=True)

def transfer(dataframe):
    distances = []
    labels = []
    for i in range(0, len(dataframe)):
        s1 = dataframe.ix[i, "sentence_A"]
        s2 = dataframe.ix[i, "sentence_B"]
        r = edit_distance(s1,s2)

        distances.append(r)
        labels.append(dataframe.ix[i, "entailment_judgment"])

    data = { "distances":distances, "labels":labels }
    return pd.DataFrame(data)


mapped_df = transfer(df1)
X = mapped_df["distances"].values
X = np.transpose(np.matrix(x_list))

Y = mapped_df["labels"].values

logreg.fit(X,Y)
# logreg.predict_proba(2)
