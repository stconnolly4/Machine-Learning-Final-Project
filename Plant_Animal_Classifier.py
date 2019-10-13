# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.image as mpimg
import cv2 # pip install opencv-python
import PIL
from PIL import Image
import IPython.display as display
import skimage
import io
import random
import sklearn
from sklearn.model_selection import train_test_split

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

felis_catus_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\felis_catus\\"
populus_trichocarpa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\populus_trichocarpa\\"




# dictionary of labels to species
class_names = {0: "populus_trichocarpa", \
               1: "felis_catus"}

# read in all paths into lists
all_felis_catus = [felis_catus_dir+"{}".format(i) for i in os.listdir(felis_catus_dir)]
all_populus_trichocarpa = [populus_trichocarpa_dir+"{}".format(i) for i in os.listdir(populus_trichocarpa_dir)]

# get number of samples
num_samples_all_felis_catus = len(all_felis_catus)
num_samples_all_populus_trichocarpa = len(all_populus_trichocarpa)

# create labels
populus_trichocarpa_labels = [0]*num_samples_all_populus_trichocarpa
felis_catus_labels = [1]*num_samples_all_felis_catus

# combine the datasets
all_samples = all_felis_catus + all_populus_trichocarpa
all_labels = felis_catus_labels + populus_trichocarpa_labels

# shuffle the data and labels in the same way
combined_samples_and_lists = list(zip(all_samples, all_labels))
random.shuffle(combined_samples_and_lists)
all_samples, all_labels = zip(*combined_samples_and_lists)

# split into test and train data
X_train, X_test, train_labels, test_labels = train_test_split(all_samples, all_labels, test_size = 0.2)

print("Done splitting")


# convert filepaths to images
train_images = []
test_images = []

def reshape_image_and_convert_to_float(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250), PIL.Image.ANTIALIAS)
    return skimage.img_as_float(img)

for image_path in X_train:
   # train_images.append(skimage.img_as_float(Image.open(image_path)))
   train_images.append(reshape_image_and_convert_to_float(image_path))

print("Done converting trainings")

for image_path in X_test:
#    test_images.append(skimage.img_as_float(Image.open(image_path)))
    test_images.append(reshape_image_and_convert_to_float(image_path))


print("Done converting testings")

# plot to test

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()



## FIGURE OUT SHAPE OF IMAGES!
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(250, 250,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
    
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("About to fit")


model.fit(np.array(train_images), np.array(train_labels), epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# cat_zero = Image.open(all_felis_catus[90])
# display.display(cat_zero)

# cat_zero_as_floats = skimage.img_as_float(cat_zero)

#plt.figure()
#plt.imshow(cat_zero_as_floats)
#plt.colorbar()
#plt.grid(False)
#plt.show()

