import numpy as np
from sklearn import linear_model

def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

trial = open('SICK_trial.txt', 'r')
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
