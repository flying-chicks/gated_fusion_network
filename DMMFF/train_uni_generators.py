# -*- encoding: utf-8 -*-

# @Time    : 1/23/19 11:27 AM
# @File    : train_uni_generators.py

from .unimodal_gan import GanForImageOrLeaf, GanForText, train
from .settings import IMAGES_SIZE, IMAGE_NOISE_SIZE
from .settings import LEAVES_SIZE, LEAF_NOISE_SIZE
from .settings import TEXTS_SIZE, TEXT_NOISE_SIZE, EMBEDDING_SIZE

from .utils import read, write
from .settings import TEXTS, LEAVES, IMAGES
from .settings import UNI_GENERATOR_FOR_IMAGE, UNI_DISCRIMINATOR_FOR_IMAGE, UNIMODAL_GAN_LOSS_FOR_IMAGE
from .settings import UNI_GENERATOR_FOR_LEAF, UNI_DISCRIMINATOR_FOR_LEAF, UNIMODAL_GAN_LOSS_FOR_LEAF
from .settings import UNI_GENERATOR_FOR_TEXT, UNI_DISCRIMINATOR_FOR_TEXT, UNIMODAL_GAN_LOSS_FOR_TEXT


TRAIN_IMAGE_GENERATOR = False
TRAIN_LEAF_GENERATOR = False
TRAIN_TEXT_GENERATOR = True

# ============================================================
# train image generators
# ============================================================
if TRAIN_IMAGE_GENERATOR:
    images = read(IMAGES)[0]
    gan4image = GanForImageOrLeaf(IMAGE_NOISE_SIZE, IMAGES_SIZE, 100)
    train(gan4image, images)
    gan4image.generator.save(UNI_GENERATOR_FOR_IMAGE)
    gan4image.discriminator.save(UNI_DISCRIMINATOR_FOR_IMAGE)
    write(gan4image.losses, UNIMODAL_GAN_LOSS_FOR_IMAGE)

# ============================================================
# train leaf generators
# ============================================================
if TRAIN_LEAF_GENERATOR:
    leaves = read(LEAVES)[0]
    gan4leaf = GanForImageOrLeaf(LEAF_NOISE_SIZE, LEAVES_SIZE, 500)
    train(gan4leaf, leaves)
    gan4leaf.generator.save(UNI_GENERATOR_FOR_LEAF)
    gan4leaf.discriminator.save(UNI_DISCRIMINATOR_FOR_LEAF)
    write(gan4leaf.losses, UNIMODAL_GAN_LOSS_FOR_LEAF)

# ============================================================
# train text generators
# ============================================================
if TRAIN_TEXT_GENERATOR:
    texts = read(TEXTS)[0]
    gan4text = GanForText(TEXT_NOISE_SIZE, TEXTS_SIZE, EMBEDDING_SIZE)
    train(gan4text, texts, 2000)
    gan4text.generator.save(UNI_GENERATOR_FOR_TEXT)
    gan4text.discriminator.save(UNI_DISCRIMINATOR_FOR_TEXT)
    write(gan4text.losses, UNIMODAL_GAN_LOSS_FOR_TEXT)