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

filtered = df[["sentence_A", "sentence_B", "entailment_judgment"]]

data = np.loadtxt(trial)




#d = edit_distance(["hello","world","my","is","John"],["hello","John","my","name","is","world"])

#print d


X = [[10],[12],[7],[9],[10],[0],[1],[2],[1],[1],[2],[3],[2],[5],[7]]
Y = ["Neu","Neu","Neu","Neu","Neu","Ent","Ent","Ent","Ent","Ent","Con","Con","Con","Con","Con",]

print len(Y)
print len(X)

logreg = linear_model.LogisticRegression()

logreg.fit(X, Y)

Z = logreg.predict([[5],[10],[1],[11],[3],[7]])
Z_proba = logreg.predict_proba([[5],[10],[1],[11],[3],[7]])

print logreg.classes_
print Z
print Z_proba
