# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 12:29 PM
# @Author  : Kennis Yu
# @File    : multimodal_gan.py

from random import sample

import numpy as np
# installed packages and modules
from keras.layers import (Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, Input, Activation, BatchNormalization,
                          concatenate, GaussianNoise, multiply, RepeatVector,
                          Lambda)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l1_l2
from numpy import zeros
from numpy.random import standard_normal

from .settings import (TEXTS_SIZE, TEXT_NOISE_SIZE, EMBEDDING_SIZE,
                       IMAGES_SIZE, IMAGE_NOISE_SIZE, LEAVES_SIZE, LEAF_NOISE_SIZE)
# created packages and modules
from .utils import Word2Embedded, tanh3


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


def generator_for_text(noise_len, embedding_len, conv_filters, conv_window_len):
    # start to create CNN
    # add input layer
    texts_noise = Input(shape=(noise_len, embedding_len), dtype="float32", name="texts_noise")

    # add first conv layer and batch-normalization layer
    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(texts_noise)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Activation(activation='relu')(hidden_layers)

    # add second conv layer and batch-normalization layer
    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Activation(activation='relu')(hidden_layers)

    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Activation(activation='relu')(hidden_layers)

    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Activation(activation='relu')(hidden_layers)

    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(hidden_layers)
    hidden_layers = BatchNormalization()(hidden_layers)
    texts_out = Activation(tanh3)(hidden_layers)

    gen_model = Model(inputs=[texts_noise], outputs=[texts_out])
    return gen_model


def generator_for_image_or_leaf(noise_len, out_len, dense_units):
    noise = Input(shape=(noise_len,), dtype="float32", name="images_noise")

    # add full-connect layer
    hidden_layers = Dense(dense_units, activation="relu")(noise)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)

    gen_out = Dense(out_len, activation="tanh")(hidden_layers)
    gen_model = Model(inputs=[noise], outputs=[gen_out])
    return gen_model


def discriminator(text_len, embedding_len, conv_filters, conv_window_len, dense_units,
                  images_size, leaves_size, lr, is_gate=True):
    # start to create CNN
    # add input layer
    texts = Input(shape=(text_len, embedding_len), dtype="float32", name="texts")
    texts_with_noise = GaussianNoise(0.01)(texts)

    # add first conv layer and max-pool layer
    texts_conv1d = Conv1D(conv_filters, conv_window_len, padding='valid',
                          activation='linear', strides=1)(texts_with_noise)
    texts_pool1d = MaxPool1D()(texts_conv1d)

    # add flatten layer
    texts_flatten = Flatten()(texts_pool1d)

    # add full-connect layer and drop-out layer
    texts_dense = Dense(10, activation="linear")(texts_flatten)
    texts_out = Dropout(0.5)(texts_dense)

    images = Input(shape=(images_size,), name='images')
    images_with_noise = GaussianNoise(0.01)(images)
    images_out = Dense(4, activation='linear')(images_with_noise)

    leaves = Input(shape=(leaves_size,), name="leaves")
    leaves_with_noise = GaussianNoise(0.01)(leaves)
    leaves_out = Dense(5, activation='linear')(leaves_with_noise)

    if is_gate:
        texts_gate = Dense(10, activation="hard_sigmoid", name='texts_gate')(concatenate([images_out, leaves_out],
                                                                                         axis=-1))
        images_gate = Dense(4, activation="hard_sigmoid", name='images_gate')(concatenate([texts_out, leaves_out],
                                                                                          axis=-1))
        leaves_gate = Dense(5, activation="hard_sigmoid", name='leaves_gate')(concatenate([texts_out, images_out],
                                                                                          axis=-1))
        texts_filtered = multiply([texts_out, texts_gate])
        images_filtered = multiply([images_out, images_gate])
        leaves_filtered = multiply([leaves_out, leaves_gate])
    else:
        texts_filtered = texts_out
        images_filtered = images_out
        leaves_filtered = leaves_out

    texts_images_kron = Kronecker([images_filtered, texts_filtered])
    texts_leaves_kron = Kronecker([leaves_filtered, texts_filtered])
    images_leaves_kron = Kronecker([images_filtered, leaves_filtered])

    datas = [texts_out, images_out, leaves_out, texts_images_kron,
                texts_leaves_kron, images_leaves_kron]

    cat_data = concatenate(datas)
    cat_hidden = Dense(dense_units, activation="linear")(cat_data)
    cat_hidden = Dropout(0.5)(cat_hidden)

    # add output layer with softmax
    cat_output = Dense(2, activation='softmax', name='cat_output',
                       activity_regularizer=l1_l2(l1=0.02, l2=0.02),
                       kernel_regularizer=l1_l2(l1=0.02, l2=0.02),
                       bias_regularizer=l1_l2(l1=0.02, l2=0.02))(cat_hidden)

    dis_model = Model(inputs=[texts, images, leaves], outputs=[cat_output])

    optimizer = RMSprop(lr=lr, clipvalue=1.0, decay=1e-8)
    dis_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return dis_model


def fix_model(model, is_trainable=False):
    model.trainable = is_trainable
    for layer in model.layers:
        layer.trainable = is_trainable


class Gan(object):
    def __init__(self):
        # create the generator and discriminator model.
        # create the generator model
        texts_size = TEXTS_SIZE
        text_noise_size = TEXT_NOISE_SIZE
        embedding_size = EMBEDDING_SIZE

        conv_filters = 300
        conv_window_len = 3

        image_size = IMAGES_SIZE
        image_noise_size = IMAGE_NOISE_SIZE
        image_dense_units = 100

        leaves_size = LEAVES_SIZE
        leaf_noise_size = LEAF_NOISE_SIZE
        leaf_dense_units = 500

        dis_lr = 1e-4
        gen_lr = 1e-3

        self.generator_for_text = generator_for_text(noise_len=text_noise_size,
                                                     embedding_len=embedding_size,
                                                     conv_filters=conv_filters,
                                                     conv_window_len=conv_window_len,
                                                     )

        self.generator_for_image = generator_for_image_or_leaf(noise_len=image_noise_size,
                                                               out_len=image_size,
                                                               dense_units=image_dense_units,
                                                               )

        self.generator_for_leaf = generator_for_image_or_leaf(noise_len=leaf_noise_size,
                                                              out_len=leaves_size,
                                                              dense_units=leaf_dense_units,
                                                              )

        # create the discriminator model
        self.discriminator = discriminator(texts_size, embedding_size,
                                           conv_filters=250,
                                           conv_window_len=3,
                                           dense_units=250,
                                           lr=dis_lr,
                                           images_size=image_size,
                                           leaves_size=leaves_size)
        # fix the discriminator
        fix_model(self.discriminator, is_trainable=False)

        # assemble the generator and discriminator model into a gan model
        text_noise_in = Input(shape=(text_noise_size, embedding_size), dtype="float32", name="text_noise_in")
        image_noise_in = Input(shape=(image_noise_size,), dtype="float32", name="image_noise_in")
        leaf_noise_in = Input(shape=(leaf_noise_size,), dtype="float32", name="leaf_noise_in")

        text_hidden_layer = self.generator_for_text(text_noise_in)
        image_hidden_layer = self.generator_for_image(image_noise_in)
        leaf_hidden_layer = self.generator_for_leaf(leaf_noise_in)

        gan_output = self.discriminator([text_hidden_layer, image_hidden_layer, leaf_hidden_layer])

        self.gan_model = Model(inputs=[text_noise_in, image_noise_in, leaf_noise_in], outputs=[gan_output])

        optimizer = RMSprop(lr=gen_lr, clipvalue=1.0, decay=1e-8)
        self.gan_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.text_len = texts_size
        self.text_noise_len = text_noise_size
        self.embedding_len = embedding_size
        self.word2embedded = Word2Embedded(texts_size)

        self.image_noise_len = image_noise_size
        self.leaf_noise_len = leaf_noise_size

        self.losses = {"gen_loss": [], "dis_loss": []}

    def train(self, texts, images, leaves, epochs=2000, batch_size=25):
        for epoch in range(epochs):
            text_seed_noises = standard_normal(size=(batch_size,
                                                     self.text_noise_len,
                                                     self.embedding_len)).astype(dtype="float32")

            image_seed_noises = standard_normal(size=(batch_size,
                                                      self.image_noise_len)).astype(dtype="float32")

            leaf_seed_noises = standard_normal(size=(batch_size,
                                                     self.leaf_noise_len)).astype(dtype="float32")

            # counterfeit text, image and leaf
            gen_embedding_mat = self.generator_for_text.predict(text_seed_noises)
            gen_image_mat = self.generator_for_image.predict(image_seed_noises)
            gen_leaf_mat = self.generator_for_leaf.predict(leaf_seed_noises)

            # sample from x with batch_size
            batch_index = sample(range(texts.shape[0]), batch_size)

            true_word_index = texts[batch_index]
            true_embedding_mat = self.word2embedded(true_word_index)

            true_image_mat = images[batch_index]
            true_leaf_mat = leaves[batch_index]

            # concatenate the counterfeit text and true text
            cat_texts = np.concatenate((true_embedding_mat, gen_embedding_mat), axis=0)
            cat_images = np.concatenate((true_image_mat, gen_image_mat), axis=0)
            cat_leaves = np.concatenate((true_leaf_mat, gen_leaf_mat), axis=0)

            target = zeros(shape=(batch_size * 2, 2), dtype="int32")
            target[:batch_size, 1] = 1
            target[batch_size:, 0] = 1

            # feed the cat data and target into discriminator
            fix_model(self.discriminator, is_trainable=True)
            dis_loss = self.discriminator.train_on_batch(x={"texts": cat_texts,
                                                            "images": cat_images,
                                                            "leaves": cat_leaves}, y=target)
            self.losses["dis_loss"].append(dis_loss[0])
            print(('epoch: {}, training discriminator, '
                   'loss: {:.2f}, accuracy: {:.2f}').format(epoch + 1, *dis_loss))

            # train Generator-Discriminator stack on input noise to non-generated output class
            text_seed_noises = standard_normal(size=(batch_size,
                                                     self.text_noise_len,
                                                     self.embedding_len)).astype(dtype="float32")

            image_seed_noises = standard_normal(size=(batch_size,
                                                      self.image_noise_len)).astype(dtype="float32")

            leaf_seed_noises = standard_normal(size=(batch_size,
                                                     self.leaf_noise_len)).astype(dtype="float32")

            target = zeros([batch_size, 2], dtype="int32")
            target[:, 1] = 1

            # train gan with discriminator fixed
            fix_model(self.discriminator, is_trainable=False)
            gen_loss = self.gan_model.train_on_batch(x={"text_noise_in": text_seed_noises,
                                                        "image_noise_in": image_seed_noises,
                                                        "leaf_noise_in": leaf_seed_noises}, y=target)
            self.losses["gen_loss"].append(gen_loss[0])
            print(("epoch: {}, training generator, "
                   "loss: {:.2f}, accuracy: {:.2f}").format(epoch + 1, *gen_loss))
            print('-' * 60)
