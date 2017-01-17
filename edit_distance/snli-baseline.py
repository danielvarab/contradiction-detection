import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

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

    m = len(s1)+1
    n = len(s2)+1

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

file_path = "SICK_train.txt" # LINK: https://raw.githubusercontent.com/ashudeep/evaluate-semantic-relatedness/master/SICK_train.txt
df = pd.read_csv(file_path, sep="\t")

equal_distribution = []
label_groups = df.groupby("entailment_judgment")
for x in label_groups.groups:
    group = label_groups.get_group(x)
    equal_distribution.append(group[:600])

df1 = pd.concat(equal_distribution, ignore_index=True)

def edit_distance_label(row):
   a = row['sentence_A']
   b = row['sentence_B']
   return edit_distance(a, b)

df1["distance"] = df1.apply (lambda row: edit_distance_label(row), axis=1)

X = df1["distance"].values
X = np.transpose(np.matrix(X))

Y = df1["entailment_judgment"].values

logreg = linear_model.LogisticRegression()
logreg.fit(X,Y)
score = logreg.score(X,Y)
print(score) #  accuracy: 0.527777777778

# logreg.predict_proba(2)

# plt.plot(X

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
