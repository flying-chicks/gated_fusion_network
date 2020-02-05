# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 3:00 PM
# @File    : settings.py

TEXTS_SIZE = 400
TEXT_NOISE_SIZE = 410
EMBEDDING_SIZE = 300

PREFIX = "/gated_fusion_network"

IMAGES_SIZE = 4
IMAGE_NOISE_SIZE = 50

LEAVES_SIZE = 300
LEAF_NOISE_SIZE = 350

EMBEDDING_MATRIX_FN = PREFIX + "/word2vec_embedding_matrix.pkl"

GFN_INIT_PARAMETERS = {
    "texts_size": TEXTS_SIZE,
    "embedding_size": EMBEDDING_SIZE,
    "vocab_size": 60230,

    "n_filter": 250,
    "kernel_size": 3,
    "hidden_size": 250,

    "images_size": IMAGES_SIZE,
    "leaves_size": LEAVES_SIZE,

    "is_gate": True,
    "is_fusion": True,
}

GFN_TRAIN_PARAMETERS = {
    "epochs": 20,
    "batch_size": 32,
}

N_GEN_SAMPLES = 15700

N_ADJUST_SAMPLES = 5000

MULTI_GENERATOR_FOR_TEXT = PREFIX + "/multimodal_gan/generator_for_text.h5"
MULTI_GENERATOR_FOR_IMAGE = PREFIX + "/multimodal_gan/generator_for_image.h5"
MULTI_GENERATOR_FOR_LEAF = PREFIX + "/multimodal_gan/generator_for_leaf.h5"
MULTI_DISCRIMINATOR = PREFIX + "/multimodal_gan/discriminator.h5"
MULTIMODAL_GAN_LOSS = PREFIX + "/multimodal_gan/losses.pkl"

UNI_GENERATOR_FOR_TEXT = PREFIX + "/unimodal_gan/generator_for_text.h5"
UNI_GENERATOR_FOR_IMAGE = PREFIX + "/unimodal_gan/generator_for_image.h5"
UNI_GENERATOR_FOR_LEAF = PREFIX + "/unimodal_gan/generator_for_leaf.h5"

UNI_DISCRIMINATOR_FOR_TEXT = PREFIX + "/unimodal_gan/discriminator_for_text.h5"
UNI_DISCRIMINATOR_FOR_IMAGE = PREFIX + "/unimodal_gan/discriminator_for_image.h5"
UNI_DISCRIMINATOR_FOR_LEAF = PREFIX + "/unimodal_gan/discriminator_for_leaf.h5"

UNIMODAL_GAN_LOSS_FOR_IMAGE = PREFIX + "/unimodal_gan/losses_for_image.pkl"
UNIMODAL_GAN_LOSS_FOR_LEAF = PREFIX + "/unimodal_gan/losses_for_leaf.pkl"
UNIMODAL_GAN_LOSS_FOR_TEXT = PREFIX + "/unimodal_gan/losses_for_text.pkl"

# pos_train, neg_train, pos_test, neg_test
TEXTS = PREFIX + "/imbalanced_data/texts.pkl"
IMAGES = PREFIX + "/imbalanced_data/images.pkl"
LEAVES = PREFIX + "/imbalanced_data/leaves.pkl"
TARGET = PREFIX + "/imbalanced_data/target.pkl"

IMBALANCED_TFIDF_MATRIX = PREFIX + "/imbalanced_data/tfidf_matrix.pkl"

# pos_train, neg_train, pos_test, neg_test
AUGMENTED_TEXTS_BY_MULTIMODAL_GAN = PREFIX + "/augmented_data_by_multimodal_gan/texts.pkl"
AUGMENTED_IMAGES_BY_MULTIMODAL_GAN = PREFIX + "/augmented_data_by_multimodal_gan/images.pkl"
AUGMENTED_LEAVES_BY_MULTIMODAL_GAN = PREFIX + "/augmented_data_by_multimodal_gan/leaves.pkl"
AUGMENTED_TARGET_BY_MULTIMODAL_GAN = PREFIX + "/augmented_data_by_multimodal_gan/target.pkl"

AUGMENTED_TEXTS_BY_UNIMODAL_GAN = PREFIX + "/augmented_data_by_unimodal_gan/texts.pkl"
AUGMENTED_IMAGES_BY_UNIMODAL_GAN = PREFIX + "/augmented_data_by_unimodal_gan/images.pkl"
AUGMENTED_LEAVES_BY_UNIMODAL_GAN = PREFIX + "/augmented_data_by_unimodal_gan/leaves.pkl"
AUGMENTED_TARGET_BY_UNIMODAL_GAN = PREFIX + "/augmented_data_by_unimodal_gan/target.pkl"

GFN_MODEL_DIR = PREFIX + "/gfn/"

OTHER_MODEL_DIR = PREFIX + "/others/"
