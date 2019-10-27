# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
#import matplotlib.image as mpimg
#import cv2 # pip install opencv-python
import PIL
from PIL import Image
#import IPython.display as display
import skimage
#import io
import random
import numpy as np
#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
#import numpy as np
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

# split images into R, G, B floats
resize_int = 250

def reshape_image_and_convert_to_float(image_path):
    img = Image.open(image_path)
    img = img.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
    return skimage.img_as_float(img)

all_images = []

for sample in all_samples:
    all_images.append(reshape_image_and_convert_to_float(sample))

all_images_R = [] 
all_images_G = [] 
all_images_B = [] 

ct = 0
for image in all_images:
    temp_R = [[0 for i in range(resize_int)] for j in range(resize_int)]
    temp_G = [[0 for i in range(resize_int)] for j in range(resize_int)]
    temp_B = [[0 for i in range(resize_int)] for j in range(resize_int)]
    for r in range(len(image)):
        row = image[r]
        for p in range(len(row)):
            pixel = row[p]
            temp_R[r][p] = pixel[0] 
            temp_G[r][p] = pixel[1]
            temp_B[r][p] = pixel[2]
    all_images_R.append(temp_R)
    all_images_G.append(temp_G)
    all_images_B.append(temp_B)

# PCA
all_images_flattened_R = []
for image in all_images_R:
    all_images_flattened_R.append(np.array(image).flatten())
  
all_images_flattened_G = []
for image in all_images_G:
    all_images_flattened_G.append(np.array(image).flatten())
    
all_images_flattened_B = []
for image in all_images_B:
    all_images_flattened_B.append(np.array(image).flatten())
  
number_of_components = 10

pca = PCA(n_components=number_of_components)
principalComponents_R = pca.fit(all_images_flattened_R)
principalComponents_G = pca.fit(all_images_flattened_G)
principalComponents_B = pca.fit(all_images_flattened_B)

# Convert all the images to PCA versions of themselves
all_images_principal_components_R = []
for image in all_images_R:
    all_images_principal_components_R.append(principalComponents_R.fit_transform(image))

all_images_principal_components_G = []
for image in all_images_G:
    all_images_principal_components_G.append(principalComponents_G.fit_transform(image))
    
all_images_principal_components_B = []
for image in all_images_B:
    all_images_principal_components_B.append(principalComponents_B.fit_transform(image))

# split into test and train data
combined_RGB = list(zip(all_images_principal_components_R, all_images_principal_components_G, all_images_principal_components_B))
X_train_RGB, X_test_RGB, train_labels, test_labels = train_test_split(combined_RGB, all_labels, test_size = 0.2)
X_train_R, X_train_G, X_train_B = zip(*X_train_RGB)
X_test_R, X_test_G, X_test_B = zip(*X_test_RGB)

# train 3 models
# R
model_R = keras.Sequential([
     keras.layers.Flatten(input_shape=(resize_int, number_of_components)),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(10, activation='softmax')
 ])
    
model_R.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
 
model_R.fit(np.array(X_train_R), np.array(train_labels), epochs=10)

# G
model_G = keras.Sequential([
     keras.layers.Flatten(input_shape=(resize_int, number_of_components)),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(10, activation='softmax')
 ])
    
model_G.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
 
model_G.fit(np.array(X_train_G), np.array(train_labels), epochs=10)

# B
model_B = keras.Sequential([
     keras.layers.Flatten(input_shape=(resize_int, number_of_components)),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(10, activation='softmax')
 ])
    
model_B.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
 
model_B.fit(np.array(X_train_R), np.array(train_labels), epochs=10)
 
# test
test_loss_R, test_acc_R = model_R.evaluate(np.array(X_test_R), np.array(test_labels), verbose=2)
test_loss_G, test_acc_G = model_R.evaluate(np.array(X_test_G), np.array(test_labels), verbose=2)
test_loss_B, test_acc_B = model_B.evaluate(np.array(X_test_B), np.array(test_labels), verbose=2)

print('\nTest accuracy R:', test_acc_R)
print('\nTest accuracy G:', test_acc_G)
print('\nTest accuracy B:', test_acc_B)

# =============================================================================

# cat_zero = Image.open(all_felis_catus[90])
# display.display(cat_zero)

# cat_zero_as_floats = skimage.img_as_float(cat_zero)

#plt.figure()
#plt.imshow(cat_zero_as_floats)
#plt.colorbar()
#plt.grid(False)
#plt.show()

