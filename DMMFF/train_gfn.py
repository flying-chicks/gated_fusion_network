# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 9:54 PM
# @File    : train_gfn.py


# installed packages and modules
from keras.utils import np_utils
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from numpy import argmax, concatenate
from time import time
from numpy.random import permutation

# created packages and modules
from .settings import GFN_INIT_PARAMETERS, GFN_TRAIN_PARAMETERS, GFN_MODEL_DIR
from .gfn import GFN
from .utils import write


def cat_data(data_sets):
    train_set = concatenate((data_sets[0], data_sets[1]), axis=0)
    test_set = concatenate((data_sets[2], data_sets[3]), axis=0)
    return train_set, test_set


def train_gfn(is_embedding_matrix, texts, images, leaves, target):
    train_texts, test_texts = cat_data(texts)
    train_images, test_images = cat_data(images)
    train_leaves, test_leaves = cat_data(leaves)
    train_target, test_target = cat_data(target)

    train_shuffle_index = permutation(range(train_texts.shape[0]))
    test_shuffle_index = permutation(range(test_texts.shape[0]))

    train_texts = train_texts[train_shuffle_index]
    test_texts = test_texts[test_shuffle_index]

    train_images = train_images[train_shuffle_index]
    test_images = test_images[test_shuffle_index]

    train_leaves = train_leaves[train_shuffle_index]
    test_leaves = test_leaves[test_shuffle_index]

    train_target = train_target[train_shuffle_index]
    test_target = test_target[test_shuffle_index]

    train_target = np_utils.to_categorical(train_target)
    test_target = np_utils.to_categorical(test_target)

    data_sets = (train_texts, test_texts, train_images, test_images,
                 train_leaves, test_leaves, train_target, test_target)

    gfn = GFN(is_embedding_matrix, **GFN_INIT_PARAMETERS)
    gfn.fit(*data_sets, **GFN_TRAIN_PARAMETERS)

    train_pred = gfn.predict(train_texts, train_images, train_leaves)
    train_target = argmax(train_target, axis=1)
    train_accuracy = accuracy_score(y_true=train_target, y_pred=train_pred)

    test_scores = gfn.predict_prob(test_texts, test_images, test_leaves)
    test_pred = argmax(test_scores, axis=1)
    test_target = argmax(test_target, axis=1)
    test_accuracy = accuracy_score(y_true=test_target, y_pred=test_pred)

    results = {"train_accuracy": train_accuracy,
               "accuracy": test_accuracy,
               "y_scores": test_scores,
               "y_true": test_target,
               "elapsed_seconds": gfn.elapsed_seconds}

    is_gate = GFN_INIT_PARAMETERS["is_gate"]
    is_fusion = GFN_INIT_PARAMETERS["is_fusion"]

    model_name = ""
    if is_gate and is_fusion:
        model_name = "gate_and_interaction_"
    if is_gate and not is_fusion:
        model_name = "gate_and_no_interaction_"
    if not is_gate and is_fusion:
        model_name = "no_gate_and_interaction_"
    if not is_gate and not is_fusion:
        model_name = "no_gate_and_no_interaction_"
    model_name += str(int(time())) + ".pkl"

    write(results, GFN_MODEL_DIR + model_name)
    print(test_accuracy)
    print(classification_report(y_true=test_target, y_pred=test_pred))
    print(confusion_matrix(y_true=test_target, y_pred=test_pred))
