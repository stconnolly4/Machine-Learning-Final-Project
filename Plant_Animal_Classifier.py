# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.image as mpimg
import cv2 # pip install opencv-python
from PIL import Image
import IPython.display as display
import skimage
import io
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

felis_catus_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\felis_catus\\"

train_felis_catus = [felis_catus_dir+"{}".format(i) for i in os.listdir(felis_catus_dir)]

cat_zero = Image.open(train_felis_catus[0])
display.display(cat_zero)

cat_zero_as_floats = skimage.img_as_float(cat_zero)

plt.figure()
plt.imshow(cat_zero_as_floats)
plt.colorbar()
plt.grid(False)
plt.show()

