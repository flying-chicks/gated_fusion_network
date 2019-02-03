# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 4:18 PM
# @File    : gfn_on_augmented_data.py


# installed packages and modules
from keras.models import load_model
from numpy.random import standard_normal
from numpy import concatenate, ones
from keras.utils.generic_utils import CustomObjectScope
from os.path import exists

# created packages and modules
from .settings import N_GEN_SAMPLES, TEXT_NOISE_SIZE, EMBEDDING_SIZE, IMAGE_NOISE_SIZE, LEAF_NOISE_SIZE
from .settings import (TEXTS, IMAGES, LEAVES, TARGET)
from .settings import TEXTS_SIZE, N_ADJUST_SAMPLES
from .utils import read, Word2Embedded, tanh3, write
from .train_gfn import train_gfn


def train_gfn_on_augmented(augmented_texts, augmented_images, augmented_leaves, augmented_target,
                           generator_for_text, generator_for_image, generator_for_leaf
                           ):
    if exists(augmented_texts) and exists(augmented_images) and exists(augmented_leaves) and exists(augmented_target):
        embedding_mat = read(augmented_texts)
        images = read(augmented_images)
        leaves = read(augmented_leaves)
        target = read(augmented_target)

    else:
        with CustomObjectScope({"tanh3": tanh3}):
            generator_for_image = load_model(generator_for_image)
            generator_for_leaf = load_model(generator_for_leaf)
            generator_for_text = load_model(generator_for_text)

        text_seed_noises = standard_normal(size=(N_GEN_SAMPLES,
                                                 TEXT_NOISE_SIZE,
                                                 EMBEDDING_SIZE)).astype(dtype="float32")
        image_seed_noises = standard_normal(size=(N_GEN_SAMPLES,
                                                  IMAGE_NOISE_SIZE)).astype(dtype="float32")
        leaf_seed_noises = standard_normal(size=(N_GEN_SAMPLES,
                                                 LEAF_NOISE_SIZE)).astype(dtype="float32")

        pos_gen_embedding_mat = generator_for_text.predict(text_seed_noises)
        pos_gen_images = generator_for_image.predict(image_seed_noises)
        pos_gen_leaves = generator_for_leaf.predict(leaf_seed_noises)

        true_texts = read(TEXTS)
        word2embedded = Word2Embedded(TEXTS_SIZE)

        pos_train_embedding_mat = concatenate((word2embedded(true_texts[0][:N_ADJUST_SAMPLES]),
                                               pos_gen_embedding_mat), axis=0)
        neg_train_embedding_mat = word2embedded(true_texts[1])
        pos_test_embedding_mat = word2embedded(concatenate((true_texts[0][N_ADJUST_SAMPLES:], true_texts[2]), axis=0))
        neg_test_embedding_mat = word2embedded(true_texts[3])

        embedding_mat = (pos_train_embedding_mat, neg_train_embedding_mat,
                         pos_test_embedding_mat, neg_test_embedding_mat)

        true_images = read(IMAGES)
        pos_train_images = concatenate((true_images[0][:N_ADJUST_SAMPLES], pos_gen_images), axis=0)
        images = (pos_train_images, true_images[1],
                  concatenate((true_images[0][N_ADJUST_SAMPLES:], true_images[2]), axis=0), true_images[3])

        true_leaves = read(LEAVES)
        pos_train_leaves = concatenate((true_leaves[0][:N_ADJUST_SAMPLES], pos_gen_leaves), axis=0)
        leaves = (pos_train_leaves, true_leaves[1],
                  concatenate((true_leaves[0][N_ADJUST_SAMPLES:], true_leaves[2]), axis=0), true_leaves[3])

        true_target = read(TARGET)
        pos_train_target = concatenate((true_target[0][:N_ADJUST_SAMPLES],
                                        ones(shape=N_GEN_SAMPLES, dtype="float32")), axis=0)
        target = (pos_train_target, true_target[1],
                  concatenate((true_target[0][N_ADJUST_SAMPLES:], true_target[2]), axis=0), true_target[3])

        write(embedding_mat, augmented_texts)
        write(images, augmented_images)
        write(leaves, augmented_leaves)
        write(target, augmented_target)

    train_gfn(True, embedding_mat, images, leaves, target)