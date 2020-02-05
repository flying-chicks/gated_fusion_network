# -*- encoding: utf-8 -*-

# @Time    : 11/24/18 8:57 PM
# @File    : gfn_on_imbalanced_data.py

# created packages and modules
from .settings import (TEXTS, IMAGES, LEAVES, TARGET)
from .utils import read
from .train_gfn import train_gfn


texts = read(TEXTS)
images = read(IMAGES)
leaves = read(LEAVES)
target = read(TARGET)

train_gfn(False, texts, images, leaves, target)