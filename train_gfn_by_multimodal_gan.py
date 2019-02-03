# -*- encoding: utf-8 -*-

# @Time    : 1/24/19 9:56 AM
# @File    : train_gfn_by_multimodal_gan.py

from .settings import (AUGMENTED_TEXTS_BY_MULTIMODAL_GAN, AUGMENTED_IMAGES_BY_MULTIMODAL_GAN,
                       AUGMENTED_LEAVES_BY_MULTIMODAL_GAN, AUGMENTED_TARGET_BY_MULTIMODAL_GAN)

from .settings import MULTI_GENERATOR_FOR_TEXT, MULTI_GENERATOR_FOR_IMAGE, MULTI_GENERATOR_FOR_LEAF

from .gfn_on_augmented_data import train_gfn_on_augmented

train_gfn_on_augmented(AUGMENTED_TEXTS_BY_MULTIMODAL_GAN, AUGMENTED_IMAGES_BY_MULTIMODAL_GAN,
                       AUGMENTED_LEAVES_BY_MULTIMODAL_GAN, AUGMENTED_TARGET_BY_MULTIMODAL_GAN,
                       MULTI_GENERATOR_FOR_TEXT, MULTI_GENERATOR_FOR_IMAGE, MULTI_GENERATOR_FOR_LEAF
                       )