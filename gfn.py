# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 2:48 PM
# @File    : gfn.py

# installed packages and modules
from keras.layers import (Input, Embedding, Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, concatenate, multiply, RepeatVector, Lambda)
from keras.models import Model
from keras.callbacks import EarlyStopping
from numpy import argmax
from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

# created packages and modules
from .utils import load_embedding_matrix
from .settings import GFN_MODEL_DIR


def kronecker_product(mat1, mat2):
    n1 = mat1.get_shape()[1]
    n2 = mat2.get_shape()[1]
    mat1 = RepeatVector(n2)(mat1)
    mat1 = concatenate([mat1[:, :, i] for i in range(n1)], axis=-1)
    mat2 = Flatten()(RepeatVector(n1)(mat2))
    result = multiply(inputs=[mat1, mat2])

    # convert (i-1)dim to i dim
    # result = Reshape((n2, n1))(result)
    return result


Kronecker = Lambda(lambda tensors: kronecker_product(tensors[0], tensors[1]))


class GFN(object):
    def __init__(self, is_embedding_matrix, texts_size, embedding_size, vocab_size,
                 n_filter, kernel_size, hidden_size, images_size,
                 leaves_size, is_gate, is_fusion):

        if is_embedding_matrix:
            _texts = Input(shape=(texts_size, embedding_size), dtype='float32', name='texts')
            embedded_texts = _texts
        else:
            embedding_matrix = load_embedding_matrix()
            _texts = Input(shape=(texts_size,), dtype='int32', name='texts')
            embedded_texts = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                                       input_length=texts_size, weights=[embedding_matrix],
                                       trainable=True)(_texts)
            embedded_texts = Dropout(0.5)(embedded_texts)

        conv_out = Conv1D(n_filter, kernel_size, padding='valid', activation='relu', strides=1)(embedded_texts)
        pooled_out = MaxPool1D()(conv_out)

        flatten_out = Flatten()(pooled_out)
        texts_hidden = Dense(hidden_size, activation="tanh")(flatten_out)
        texts_hidden = Dropout(0.5)(texts_hidden)

        texts_hidden = Dense(hidden_size, activation="tanh")(texts_hidden)
        texts_hidden = Dropout(0.5)(texts_hidden)
        texts_out = Dense(100, activation='tanh', name='texts_out')(texts_hidden)

        _images = Input(shape=(images_size,), name='images')
        images_out = Dense(4, activation='tanh', name='images_out')(_images)

        _leaves = Input(shape=(leaves_size,), name="leaves")
        leaves_hidden = Dense(250, activation="tanh")(_leaves)
        leaves_hidden = Dropout(0.5)(leaves_hidden)
        # leaves_hidden = Dense(250, activation="tanh")(leaves_hidden)
        # leaves_hidden = Dropout(0.5)(leaves_hidden)
        leaves_out = Dense(100, activation='tanh', name='leaves_out')(leaves_hidden)

        if is_gate:
            texts_gate = Dense(100, activation="sigmoid", name='texts_gate')(concatenate([images_out, leaves_out],
                                                                                         axis=-1))
            images_gate = Dense(4, activation="sigmoid", name='images_gate')(concatenate([texts_out, leaves_out],
                                                                                         axis=-1))
            leaves_gate = Dense(100, activation="sigmoid", name='leaves_gate')(concatenate([texts_out, images_out],
                                                                                           axis=-1))
            texts_filtered = multiply([texts_out, texts_gate])
            images_filtered = multiply([images_out, images_gate])
            leaves_filtered = multiply([leaves_out, leaves_gate])

        if is_fusion:
            texts_images_kron = Kronecker([images_filtered, texts_filtered])
            texts_leaves_kron = Kronecker([leaves_filtered, texts_filtered])
            images_leaves_kron = Kronecker([images_filtered, leaves_filtered])
            # images_leaves_texts_kron = Kronecker([images_out, texts_leaves_kron])
            # datas = [texts_out, images_out, leaves_out, texts_images_kron,
            #          texts_leaves_kron, images_leaves_kron, images_leaves_texts_kron]
            datas = [texts_out, images_out, leaves_out, texts_images_kron,
                     texts_leaves_kron, images_leaves_kron]
        else:
            datas = [texts_out, images_out, leaves_out]

        cat_data = concatenate(datas)
        # cat_hidden = Dropout(0.5)(cat_data)

        cat_hidden = Dense(1000, activation="tanh")(cat_data)
        cat_hidden = Dropout(0.5)(cat_hidden)
        # cat_hidden = Dense(300, activation="tanh")(cat_hidden)
        # cat_hidden = Dropout(0.5)(cat_hidden)

        cat_out = Dense(2, activation='sigmoid', name='cat_out')(cat_hidden)

        self._model = Model(inputs=[_texts, _images, _leaves], outputs=[cat_out])

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy', metrics=['accuracy'])
        self.elapsed_seconds = 0

    def fit(self, train_texts, test_texts, train_images, test_images, train_leaves, test_leaves,
            train_target, test_target, epochs, batch_size):
        start = time()
        early_stopping = EarlyStopping(monitor="val_loss", patience=1, mode="min", min_delta=0.0001)
        self._model.fit(x={'texts': train_texts, 'images': train_images, 'leaves': train_leaves},
                        y={'cat_out': train_target},
                        validation_data=({'texts': test_texts, 'images': test_images, 'leaves': test_leaves},
                                         {'cat_out': test_target}),
                        shuffle=True,
                        callbacks=[early_stopping],
                        epochs=epochs,
                        batch_size=batch_size
                        )
        end = time()
        self.elapsed_seconds = end - start

    def predict_prob(self, texts, images, leaves):
        return self._model.predict(x={'texts': texts, 'images': images, 'leaves': leaves})

    def predict(self, texts, images, leaves):
        return argmax(self.predict_prob(texts, images, leaves), axis=1)

    def save(self):
        self._model.save(GFN_MODEL_DIR + "model_" + str(int(time())))
