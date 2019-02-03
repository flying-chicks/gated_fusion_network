# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 12:32 PM
# @File    : utils.py

from sklearn.externals import joblib
from keras.layers import Input, Embedding
from keras.models import Model
from sklearn.preprocessing import minmax_scale
from numpy import asarray
from keras.backend import tanh
from re import compile
from numpy import argmax, mean, isnan
from sklearn.metrics import classification_report
from prettytable import PrettyTable
from os import listdir
from collections import defaultdict

from .settings import EMBEDDING_MATRIX_FN


def read(fname):
    with open(fname, 'rb') as fr:
        return joblib.load(fr)


def write(obj, fname):
    with open(fname, 'wb') as fw:
        joblib.dump(obj, fw)


def load_embedding_matrix():
    embedding_matrix = asarray(read(EMBEDDING_MATRIX_FN), dtype="float32")
    return embedding_matrix


class Word2Embedded(object):
    def __init__(self, text_len):
        embedding_matrix = load_embedding_matrix()
        text_input = Input(shape=(text_len, ), dtype="int32")
        embedded = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                             input_length=text_len, weights=[embedding_matrix], trainable=False)(text_input)
        self.model = Model(inputs=[text_input], outputs=[embedded])

    def __call__(self, texts):
        """

        :type texts: list.
        """
        return self.model.predict(texts)


def tanh3(x):
    return 3 * tanh(x)


def cast_data(pos_train_index, neg_train_index, pos_test_index, neg_test_index, dataset,
              fname, is_int, is_standardize, is_ravel):

    pos_train = dataset.loc[pos_train_index, :]
    neg_train = dataset.loc[neg_train_index, :]
    pos_test = dataset.loc[pos_test_index, :]
    neg_test = dataset.loc[neg_test_index, :]

    data_sets = [pos_train, neg_train, pos_test, neg_test]
    dealts = []
    for d in data_sets:
        if is_int:
            dealt = asarray(d, "int32")
        else:
            dealt = asarray(d, "float32")

        if is_standardize:
            dealt = minmax_scale(dealt, feature_range=(-1, 1), axis=0)

        if is_ravel:
            dealt = dealt.ravel()
        dealts.append(dealt)

    write(tuple(dealts), fname)


def fprint_result(results):
    train_accuracies = []
    accuracies = []
    elapsed = []
    precisions = []
    recalls = []
    pattern = compile("total\s+?([0-9\.]+?)\s+?([0-9\.]+?)\s+?[0-9\.]+?")
    for result in results:
        if "train_accuracy" in result:
            train_accuracies.append(result["train_accuracy"])

        accuracies.append(result["accuracy"])
        elapsed.append(result["elapsed_seconds"])
        report = classification_report(result["y_true"], argmax(result["y_scores"], axis=1))
        pr = pattern.findall(report)
        precisions.append(float(pr[0][0]))
        recalls.append(float(pr[0][1]))

    avg_train_accuracy = round(mean(train_accuracies), 4)
    avg_accuracy = round(mean(accuracies), 4)
    avg_elapsed = round(mean(elapsed) / 60, 4)
    avg_precision = round(mean(precisions), 4)
    avg_recall = round(mean(recalls), 4)
    f1_score = round(2 * avg_precision * avg_recall / (avg_precision + avg_recall), 4)

    table = (PrettyTable(field_names=["precision", "recall", "f1-score", "accuracy", "time"])
             if isnan(avg_train_accuracy) else
             PrettyTable(field_names=["precision", "recall", "f1-score", "train_accuracy", "accuracy", "time"]))
    table.align["name"] = "l"
    table.padding_width = 1
    table.add_row([avg_precision, avg_recall, f1_score, avg_train_accuracy, avg_accuracy, avg_elapsed])
    print(table)


def format_result(prefix):
    results = defaultdict(list)
    pattern = compile("([a-z_]+?)\_[0-9]+?")
    for f in listdir(prefix):
        name = pattern.findall(f)[0]
        results[name].append(read(prefix + "/" + f))

    for k in results:
        print(k)
        fprint_result(results[k])
