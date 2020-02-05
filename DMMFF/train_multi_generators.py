# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 4:54 PM
# @File    : train_multi_generators.py

from .utils import read, write
from .multimodal_gan import Gan
from .settings import (MULTI_GENERATOR_FOR_IMAGE, MULTI_GENERATOR_FOR_LEAF,
                       MULTI_GENERATOR_FOR_TEXT, MULTI_DISCRIMINATOR, MULTIMODAL_GAN_LOSS,
                       TEXTS, LEAVES, IMAGES)

texts = read(TEXTS)[0]

images = read(IMAGES)[0]

leaves = read(LEAVES)[0]

gan = Gan()
gan.train(texts, images, leaves, epochs=3000, batch_size=50)

gan.generator_for_text.save(MULTI_GENERATOR_FOR_TEXT)
gan.generator_for_image.save(MULTI_GENERATOR_FOR_IMAGE)
gan.generator_for_leaf.save(MULTI_GENERATOR_FOR_LEAF)
gan.discriminator.save(MULTI_DISCRIMINATOR)
write(gan.losses, MULTIMODAL_GAN_LOSS)