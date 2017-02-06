import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import os
import sys

sys.path.insert(0, os.path.abspath('..'))


# Custom modules from Git repo
import scoring.edit_distance.edit_distance as EDIT_DISTANCE
import util.util as UTIL
import embedding.glove.glove as GLOVE


def test_edit_distance_with_glove():
    corpus = UTIL.read_lines("../datasets/glove_data/snli_sentenceA_72k_train.txt")
    synonyms = UTIL.read_lines("../datasets/glove_data/synonym.txt")
    antonyms = UTIL.read_lines("../datasets/glove_data/antonym.txt")
    path_to_model = "../tmp/"

    W = GLOVE.embed(corpus, synonyms, antonyms, path_to_model, save_often=True)

    # Merge and normalize word vectors
    W = UTIL.merge_main_context(W)

    test_similarity()
    test_dissimilarity()
    test_subcost('boy','woman')
    test_subcost('boy','walking')
    test_subcost('tall','small')
    test_subcost('small','tall')
    test_subcost('boy','father')


def test_edit_distance_with_sick():
    file_path = "SICK_train.txt"  # LINK: https://raw.githubusercontent.com/ashudeep/evaluate-semantic-relatedness/master/SICK_train.txt
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
        return EDIT_DISTANCE.edit_distance(a, b)

    df1["distance"] = df1.apply(lambda row: edit_distance_label(row), axis=1)

    X = df1["distance"].values
    X = np.transpose(np.matrix(X))

    Y = df1["entailment_judgment"].values

    logreg = linear_model.LogisticRegression()
    logreg.fit(X, Y)
    score = logreg.score(X, Y)
    print(score)  # accuracy: 0.527777777778


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

