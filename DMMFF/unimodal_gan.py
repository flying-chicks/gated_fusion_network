# -*- encoding: utf-8 -*-

# @Time    : 1/22/19 4:29 PM
# @File    : unimodal_gan.py

from keras.layers import (Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, Input, GaussianNoise)
from keras.models import Model
from keras.optimizers import RMSprop
from numpy.random import standard_normal
from numpy import zeros, concatenate as np_concate
from random import sample
from keras.regularizers import l1_l2


from .multimodal_gan import generator_for_image_or_leaf, generator_for_text, fix_model, Word2Embedded


def discriminator_for_image_or_leaf(inputs_size, dense_units, lr):
    inputs = Input(shape=(inputs_size,), name='inputs')
    inputs_with_noise = GaussianNoise(0.01)(inputs)
    hidden_layer = Dense(dense_units, activation="sigmoid")(inputs_with_noise)
    hidden_layer = Dropout(0.5)(hidden_layer)

    hidden_layer = Dense(dense_units, activation="sigmoid")(hidden_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    outputs = Dense(2, activation="softmax", name="outputs",
                    activity_regularizer=l1_l2(l1=0, l2=0.02),
                    kernel_regularizer=l1_l2(l1=0, l2=0.02),
                    bias_regularizer=l1_l2(l1=0, l2=0.02))(hidden_layer)

    dis_model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = RMSprop(lr=lr, clipvalue=1.0, decay=1e-8)
    dis_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return dis_model


def discriminator_for_text(text_len, embedding_len, conv_filters, conv_window_len, dense_units, lr):
    # start to create CNN
    # add input layer
    texts = Input(shape=(text_len, embedding_len), dtype="float32", name="texts")

    # add drop-out layer
    hidden_layers = GaussianNoise(0.01)(texts)

    # add first conv layer and max-pool layer
    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid',
                           activation='sigmoid', strides=1)(hidden_layers)
    hidden_layers = MaxPool1D()(hidden_layers)

    # add flatten layer
    hidden_layers = Flatten()(hidden_layers)

    hidden_layers = Dense(dense_units, activation="sigmoid")(hidden_layers)
    hidden_layers = Dropout(0.5)(hidden_layers)

    outputs = Dense(2, activation="softmax", name="outputs",
                    activity_regularizer=l1_l2(l1=0, l2=0.02),
                    kernel_regularizer=l1_l2(l1=0, l2=0.02),
                    bias_regularizer=l1_l2(l1=0, l2=0.02))(hidden_layers)

    dis_model = Model(inputs=[texts], outputs=[outputs])
    optimizer = RMSprop(lr=lr, clipvalue=1.0, decay=1e-8)
    dis_model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return dis_model


class GanForImageOrLeaf(object):
    def __init__(self, noise_size, inputs_size, dense_units):
        gen_lr = 1e-3
        dis_lr = 1e-4

        self.generator = generator_for_image_or_leaf(noise_len=noise_size,
                                                     out_len=inputs_size,
                                                     dense_units=dense_units)

        # create the discriminator model
        self.discriminator = discriminator_for_image_or_leaf(inputs_size, dense_units, dis_lr)

        # fix the discriminator
        fix_model(self.discriminator, is_trainable=False)

        # assemble the generator and discriminator model into a gan model
        noise_in = Input(shape=(noise_size,), dtype="float32", name="noise_in")
        gen_outputs = self.generator(noise_in)
        gan_outputs = self.discriminator(gen_outputs)
        self.gan_model = Model(inputs=[noise_in], outputs=[gan_outputs])

        optimizer = RMSprop(lr=gen_lr, clipvalue=1.0, decay=1e-8)
        self.gan_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.noise_size = noise_size
        self.losses = {"gen_loss": [], "dis_loss": []}


class GanForText(object):
    def __init__(self, noise_size, text_size, embedding_size):
        conv_filters = 300
        conv_window_len = 3
        gen_lr = 1e-3
        dis_lr = 1e-4
        dens_units = 500

        self.generator = generator_for_text(noise_len=noise_size,
                                            embedding_len=embedding_size,
                                            conv_filters=conv_filters,
                                            conv_window_len=conv_window_len)

        # create the discriminator model
        self.discriminator = discriminator_for_text(text_size, embedding_size, conv_filters,
                                                    conv_window_len, dens_units, dis_lr)

        # fix the discriminator
        fix_model(self.discriminator, is_trainable=False)

        # assemble the generator and discriminator model into a gan model
        noise_in = Input(shape=(noise_size, embedding_size), dtype="float32", name="noise_in")
        gen_outputs = self.generator(noise_in)
        gan_outputs = self.discriminator(gen_outputs)
        self.gan_model = Model(inputs=[noise_in], outputs=[gan_outputs])

        optimizer = RMSprop(lr=gen_lr, clipvalue=1.0, decay=1e-8)
        self.gan_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.word2embedded = Word2Embedded(text_size)

        self.noise_size = noise_size
        self.embedding_size = embedding_size
        self.losses = {"gen_loss": [], "dis_loss": []}


def train(gan, inputs, epochs=4000, batch_size=32):
    for epoch in range(epochs):
        # ==========================================================================================================
        # train discriminator
        # ==========================================================================================================
        if hasattr(gan, "embedding_size"):
            seed_noises = standard_normal(size=(batch_size, gan.noise_size,
                                                gan.embedding_size)).astype(dtype="float32")
        else:
            seed_noises = standard_normal(size=(batch_size, gan.noise_size)).astype(dtype="float32")
        fake_samples = gan.generator.predict(seed_noises)

        batch_index = sample(range(inputs.shape[0]), batch_size)
        if hasattr(gan, "word2embedded"):
            real_samples = gan.word2embedded(inputs[batch_index])
        else:
            real_samples = inputs[batch_index]

        cat_samples = np_concate((real_samples, fake_samples), axis=0)

        target = zeros(shape=(batch_size * 2, 2), dtype="int32")
        target[:batch_size, 1] = 1
        target[batch_size:, 0] = 1

        # feed the concatenated samples and corresponding target into discriminator
        fix_model(gan.discriminator, is_trainable=True)

        dis_loss = gan.discriminator.train_on_batch(x=cat_samples, y=target)
        gan.losses["dis_loss"].append(dis_loss[0])

        print(('epoch: {}, training discriminator, '
               'loss: {:.2f}, accuracy: {:.2f}').format(epoch + 1, *dis_loss))

        # ======================================================================================================
        # train generator
        # ======================================================================================================
        if hasattr(gan, "embedding_size"):
            seed_noises = standard_normal(size=(batch_size, gan.noise_size,
                                                gan.embedding_size)).astype(dtype="float32")
        else:
            seed_noises = standard_normal(size=(batch_size, gan.noise_size)).astype(dtype="float32")
        target = zeros([batch_size, 2], dtype="int32")
        target[:, 1] = 1

        # train gan with discriminator fixed
        fix_model(gan.discriminator, is_trainable=False)
        gen_loss = gan.gan_model.train_on_batch(x=seed_noises, y=target)
        gan.losses["gen_loss"].append(gen_loss[0])
        print(("epoch: {}, training generator, "
               "loss: {:.2f}, accuracy: {:.2f}").format(epoch + 1, *gen_loss))
        print('-' * 60)
