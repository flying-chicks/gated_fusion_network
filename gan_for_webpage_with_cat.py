from keras.layers import (Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, Embedding, Input, Activation, BatchNormalization,
                          concatenate, GaussianNoise)
from keras.models import Model
from keras.optimizers import RMSprop
from numpy.random import standard_normal
from numpy import zeros
import numpy as np
from random import sample
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from keras.regularizers import l1_l2

from warnings import filterwarnings
filterwarnings("ignore")


def read(fname):
    with open(fname, 'rb') as fr:
        return joblib.load(fr)


def generator_for_text(noise_len, embedding_len, conv_filters, conv_window_len):
    # start to create CNN
    # add input layer
    texts_noise = Input(shape=(noise_len, embedding_len), dtype="float32", name="texts_noise")

    # add first conv layer and batchnormlization layer
    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid', strides=1)(texts_noise)
    hidden_layers = BatchNormalization()(hidden_layers)
    hidden_layers = Activation(activation='relu')(hidden_layers)

    # add second conv layer and batchnormlization layer
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
    texts_out = Activation("tanh")(hidden_layers)

    gen_model = Model(inputs=[texts_noise], outputs=[texts_out])
    return gen_model


def generator_for_image_or_leaf(noise_len, out_len, dense_units):
    noise = Input(shape=(noise_len, ), dtype="float32", name="images_noise")

    # add full-connect layer
    hidden_layers = Dense(dense_units, activation="relu")(noise)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)
    hidden_layers = Dense(dense_units, activation="relu")(hidden_layers)

    gen_out = Dense(out_len, activation="tanh")(hidden_layers)
    gen_model = Model(inputs=[noise], outputs=[gen_out])
    return gen_model


def discriminator(text_len, embedding_len, conv_filters, conv_window_len, dense_units,
                  images_size, leaves_size, lr):
    # start to create CNN
    # add input layer
    texts = Input(shape=(text_len, embedding_len), dtype="float32", name="texts")

    # add drop-out layer
    hidden_layers = GaussianNoise(0.01)(texts)
    # hidden_layers = Dropout(0.5)(texts)

    # add first conv layer and max-pool layer
    hidden_layers = Conv1D(conv_filters, conv_window_len, padding='valid',
                           activation='sigmoid', strides=1)(hidden_layers)
    hidden_layers = MaxPool1D()(hidden_layers)

    # add flatten layer
    hidden_layers = Flatten()(hidden_layers)

    # add full-connect layer and drop-out layer
    hidden_layers = Dense(dense_units, activation="sigmoid")(hidden_layers)
    hidden_layers = Dropout(0.5)(hidden_layers)
    # hidden_layers = Dense(dense_units, activation="sigmoid")(hidden_layers)
    # hidden_layers = Dropout(0.5)(hidden_layers)

    images = Input(shape=(images_size,), name='images')
    images_with_noise = GaussianNoise(0.01)(images)

    leaves = Input(shape=(leaves_size,), name="leaves")
    leaves_with_noise = GaussianNoise(0.01)(leaves)

    cat_data = concatenate([hidden_layers, images_with_noise, leaves_with_noise], axis=-1)

    cat_hidden = Dense(dense_units, activation="sigmoid")(cat_data)
    cat_hidden = Dropout(0.5)(cat_hidden)
    cat_hidden = Dense(dense_units, activation="sigmoid")(cat_hidden)
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


class Word2Embedded(object):
    def __init__(self, text_len, embedding_matrix):
        text_input = Input(shape=(text_len, ), dtype="int32")
        embedded = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                             input_length=text_len, weights=[embedding_matrix], trainable=False)(text_input)
        self.model = Model(inputs=[text_input], outputs=[embedded])

    def __call__(self, texts):
        return self.model.predict(texts)


class Gan(object):
    def __init__(self):
        # create the generator and discriminator model.
        # create the generator model
        text_len = 400
        text_noise_len = 410
        embedding_len = 300
        embedding_matrix = read("/home/ygy/embedding_matrix.pkl").astype("float32")

        # map data into [-1, 1] corresponding to the tanh activation of generator's output layer.
        embedding_matrix = minmax_scale(embedding_matrix, feature_range=(-1, 1), axis=1)

        conv_filters = 300
        conv_window_len = 3

        image_len = 4
        image_noise_len = 50
        image_dense_units = 100

        leaf_len = 300
        leaf_noise_len = 350
        leaf_dense_units = 500

        dis_lr = 1e-4
        gen_lr = 1e-3

        self.generator_for_text = generator_for_text(noise_len=text_noise_len,
                                                     embedding_len=embedding_len,
                                                     conv_filters=conv_filters,
                                                     conv_window_len=conv_window_len,
                                                     )

        self.generator_for_image = generator_for_image_or_leaf(noise_len=image_noise_len,
                                                               out_len=image_len,
                                                               dense_units=image_dense_units,
                                                               )

        self.generator_for_leaf = generator_for_image_or_leaf(noise_len=leaf_noise_len,
                                                              out_len=leaf_len,
                                                              dense_units=leaf_dense_units,
                                                              )

        # create the discriminator model
        self.discriminator = discriminator(text_len, embedding_len,
                                           conv_filters=250,
                                           conv_window_len=3,
                                           dense_units=250,
                                           lr=dis_lr,
                                           images_size=image_len,
                                           leaves_size=leaf_len)
        # fix the discriminator
        fix_model(self.discriminator, is_trainable=False)

        # ensamble the generator and discriminator model into a gan model
        text_noise_in = Input(shape=(text_noise_len, embedding_len), dtype="float32", name="text_noise_in")
        image_noise_in = Input(shape=(image_noise_len,), dtype="float32", name="image_noise_in")
        leaf_noise_in = Input(shape=(leaf_noise_len,), dtype="float32", name="leaf_noise_in")

        text_hidden_layer = self.generator_for_text(text_noise_in)
        image_hidden_layer = self.generator_for_image(image_noise_in)
        leaf_hidden_layer = self.generator_for_leaf(leaf_noise_in)

        gan_output = self.discriminator([text_hidden_layer, image_hidden_layer, leaf_hidden_layer])

        self.gan_model = Model(inputs=[text_noise_in, image_noise_in, leaf_noise_in], outputs=[gan_output])

        optimizer = RMSprop(lr=gen_lr, clipvalue=1.0, decay=1e-8)
        self.gan_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        self.text_len = text_len
        self.text_noise_len = text_noise_len
        self.embedding_len = embedding_len
        self.word2embedded = Word2Embedded(text_len, embedding_matrix)

        self.image_noise_len = image_noise_len
        self.leaf_noise_len = leaf_noise_len

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


if __name__ == "__main__":

    texts_ = read("/home/ygy/concat_sequences.pkl")
    old_data = read("/home/ygy/features.pkl")
    target_ = read("/home/ygy/label.pkl")

    # images_ = old_data.iloc[:, 200:204]
    images_ = read('/home/ygy/images_fixed.pkl')
    leaves_ = old_data.iloc[:, 204:]

    index = set(texts_.index) & set(images_.index) & set(leaves_.index) & set(target_.index[target_.iloc[:, 0] == 1])

    texts_ = texts_.loc[~texts_.index.duplicated(), :]
    texts_ = texts_.loc[index, :].values.astype("float32")

    images_ = images_.loc[~images_.index.duplicated(), :]
    images_ = minmax_scale(images_.loc[index, :].values.astype("float32"), feature_range=(-1, 1), axis=0)

    leaves_ = leaves_.loc[~leaves_.index.duplicated(), :]
    leaves_ = minmax_scale(leaves_.loc[index, :].values.astype("float32"), feature_range=(-1, 1), axis=0)

    # generator based on cnn
    gan = Gan()
    gan.train(texts_, images_, leaves_, epochs=5000, batch_size=50)

    gan.generator_for_text.save('/home/ygy/gan_model/generator_for_text.h5')
    gan.generator_for_image.save('/home/ygy/gan_model/generator_for_image.h5')
    gan.generator_for_leaf.save('/home/ygy/gan_model/generator_for_leaf.h5')
    gan.discriminator.save('/home/ygy/gan_model/discriminator.h5')
    with open('/home/ygy/gan_model/losses.pkl', 'wb') as fw:
        joblib.dump(gan.losses, fw)
