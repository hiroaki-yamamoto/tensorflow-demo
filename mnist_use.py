#!/usr/bin/env python
# coding=utf-8

"""MNIST model use."""

from glob import glob

from keras import models
import numpy as np
from PIL import Image, ImageOps

model = models.load_model("mnist_cnn.h5")

for fpath in glob("actual_data/*.png"):
    img = ImageOps.invert(Image.open(fpath))
    img = np.array(img.getdata()).reshape(1, 28, 28, 1).astype('float32') / 255
    prob = list(model.predict(img, batch_size=128)[0])
    print(f"fpath: {fpath}, Perhaps this?: {prob.index(max(prob))}")
