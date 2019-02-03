# -*- encoding: utf-8 -*-

# @Time    : 11/29/18 9:15 AM
# @File    : others_on_imbalanced_data.py


from numpy import concatenate
from numpy.random import permutation
from os.path import exists
# from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .utils import read, write
from .settings import TEXTS, IMBALANCED_TFIDF_MATRIX, IMAGES, LEAVES, TARGET, OTHER_MODEL_DIR
from .convert2tfidf import convert2tfidf


if not exists(IMBALANCED_TFIDF_MATRIX):
    texts = read(TEXTS)
    target = read(TARGET)
    tfidf_matrix = convert2tfidf(texts, target, IMBALANCED_TFIDF_MATRIX)
else:
    tfidf_matrix = read(IMBALANCED_TFIDF_MATRIX)

target = read(TARGET)
train_target = concatenate((target[0], target[1]), axis=0)
test_target = concatenate((target[2], target[3]), axis=0)

train_shuffle_index = permutation(range(train_target.shape[0]))
test_shuffle_index = permutation(range(test_target.shape[0]))
train_target = train_target[train_shuffle_index]
test_target = test_target[test_shuffle_index]

images = read(IMAGES)
leaves = read(LEAVES)

pos_train_cat = concatenate((tfidf_matrix[0], images[0], leaves[0]), axis=1)
neg_train_cat = concatenate((tfidf_matrix[1], images[1], leaves[1]), axis=1)
train_cat = concatenate((pos_train_cat, neg_train_cat), axis=0)
train_cat = train_cat[train_shuffle_index]

pos_test_cat = concatenate((tfidf_matrix[2], images[2], leaves[2]), axis=1)
neg_test_cat = concatenate((tfidf_matrix[3], images[3], leaves[3]), axis=1)
test_cat = concatenate((pos_test_cat, neg_test_cat), axis=0)
test_cat = test_cat[test_shuffle_index]

# clf = SVC(probability=True, gamma=0.01)
clf = MLPClassifier(hidden_layer_sizes=(2000, 2000),
                    activation="logistic",
                    batch_size=32,
                    max_iter=500,
                    verbose=True)
# clf = DecisionTreeClassifier()

start = time()
clf.fit(train_cat, train_target)

end = time()
elapsed_seconds = end - start

pred = clf.predict(test_cat)
pred_prob = clf.predict_proba(test_cat)
accuracy = accuracy_score(y_true=test_target, y_pred=pred)
print(accuracy)
print(confusion_matrix(y_true=test_target, y_pred=pred))
print(classification_report(y_true=test_target, y_pred=pred))

write({"accuracy": accuracy, "elapsed_seconds": elapsed_seconds,
       "y_scores": pred_prob, "y_true": test_target},
      OTHER_MODEL_DIR + "mlp_on_imbalanced_data_{}.pkl".format(int(time())))