# -*- encoding: utf-8 -*-

# @Time    : 1/24/19 9:57 AM
# @File    : train_gfn_by_unimodal_gan.py

from .settings import (AUGMENTED_TEXTS_BY_UNIMODAL_GAN, AUGMENTED_IMAGES_BY_UNIMODAL_GAN,
                       AUGMENTED_LEAVES_BY_UNIMODAL_GAN, AUGMENTED_TARGET_BY_UNIMODAL_GAN)

from .settings import UNI_GENERATOR_FOR_TEXT, UNI_GENERATOR_FOR_IMAGE, UNI_GENERATOR_FOR_LEAF

from .gfn_on_augmented_data import train_gfn_on_augmented

train_gfn_on_augmented(AUGMENTED_TEXTS_BY_UNIMODAL_GAN, AUGMENTED_IMAGES_BY_UNIMODAL_GAN,
                       AUGMENTED_LEAVES_BY_UNIMODAL_GAN, AUGMENTED_TARGET_BY_UNIMODAL_GAN,
                       UNI_GENERATOR_FOR_TEXT, UNI_GENERATOR_FOR_IMAGE, UNI_GENERATOR_FOR_LEAF
                       )