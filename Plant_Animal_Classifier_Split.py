from __future__ import absolute_import, division, print_function, unicode_literals
import os
# import matplotlib.image as mpimg
# import cv2 # pip install opencv-python
import PIL
from PIL import Image
import IPython.display as display
import skimage
# import io
import random
import numpy as np
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
# import numpy as np
import matplotlib.pyplot as plt


class Plant_animal_Classifier:
    def __init__(self, class_name, plant_image_dir, animal_image_dir, resize_int=50):

        self.class_name = class_name
        self.Pimages = plant_image_dir
        self.Aimages = animal_image_dir
        self.resize_int = resize_int

    def main_loop(self):
        all_samples, all_labels = self.split_categorically()

        all_images, all_images_R, all_images_G, all_images_B =\
            self.create_lists(all_samples, self.resize_int)

        all_PCA_R , all_PCA_G, all_PCA_B =\
            self.pca_alg(all_images, self.resize_int, all_images_R, all_images_G, all_images_B)

        X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B, test_labels, train_labels = \
            self.split_train_rgb(all_PCA_R, all_PCA_G, all_PCA_B, all_labels)

        X_train, X_test = self.create_test_train_lists(X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B)

        test_loss, test_acc = self.train_compile_model(X_train, X_test, train_labels, test_labels)

        self.get_results(test_loss, test_acc)

    def get_results(self, test_lost, test_acc):
        print('\nTest accuracy:', test_acc)


    def train_compile_model(self, X_train, X_test, train_labels, test_labels):
        # train model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(np.array(X_train), np.array(train_labels), epochs=10)

        test_loss, test_acc = model.evaluate(np.array(X_test), np.array(test_labels), verbose=2)

        return test_loss, test_acc

    def create_test_train_lists(self, X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B):
        X_train = []
        for i in range(len(X_train_R)):
            X_train.append(np.concatenate((X_train_R[i], X_train_G[i], X_train_B[i])))

        X_test = []
        for i in range(len(X_test_R)):
            X_test.append(np.concatenate((X_test_R[i], X_test_G[i], X_test_B[i])))

        return X_train, X_test

    def split_train_rgb(self, all_images_principal_components_R, all_images_principal_components_G, all_images_principal_components_B, all_labels):
        combined_RGB = list(
            zip(all_images_principal_components_R, all_images_principal_components_G,
                all_images_principal_components_B))
        X_train_RGB, X_test_RGB, train_labels, test_labels = train_test_split(combined_RGB, all_labels, test_size=0.2)
        X_train_R, X_train_G, X_train_B = zip(*X_train_RGB)
        X_test_R, X_test_G, X_test_B = zip(*X_test_RGB)

        return X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B, test_labels, train_labels


    def pca_alg(self, all_images, resize_int, all_images_R, all_images_G, all_images_B):
        all_images_flattened_R = np.zeros((len(all_images), resize_int * resize_int))
        for i in range(len(all_images_R)):
            image = all_images_R[i]
            all_images_flattened_R[i] = np.array(image).flatten()

        all_images_flattened_G = []
        for image in all_images_G:
            all_images_flattened_G.append(np.array(image).flatten())

        all_images_flattened_B = []
        for image in all_images_B:
            all_images_flattened_B.append(np.array(image).flatten())

        number_of_components = 10

        pca_R = PCA(n_components=number_of_components)
        pca_G = PCA(n_components=number_of_components)
        pca_B = PCA(n_components=number_of_components)

        all_images_principal_components_R = pca_R.fit_transform(all_images_flattened_R)
        all_images_principal_components_G = pca_G.fit_transform(all_images_flattened_G)
        all_images_principal_components_B = pca_B.fit_transform(all_images_flattened_B)

        return all_images_principal_components_R, all_images_principal_components_G, all_images_principal_components_B

    def create_lists(self, all_samples, resize_int):
        all_images = []
        for sample in all_samples:
            all_images.append(self.reshape_image_and_convert_to_float(sample, resize_int))
        all_images_R = []
        all_images_G = []
        all_images_B = []

        for image in all_images:
            #    temp_R = [[0 for i in range(resize_int)] for j in range(resize_int)]
            #    temp_G = [[0 for i in range(resize_int)] for j in range(resize_int)]
            #    temp_B = [[0 for i in range(resize_int)] for j in range(resize_int)]
            temp_R = np.zeros((resize_int, resize_int))
            temp_G = np.zeros((resize_int, resize_int))
            temp_B = np.zeros((resize_int, resize_int))
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

        return all_images, all_images_R, all_images_G, all_images_B

    def reshape_image_and_convert_to_float(self, image_path, resize_int):
        img = Image.open(image_path)
        img = img.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
        return skimage.img_as_float(img)

    def split_categorically(self):
        # read in all paths into lists
        all_animals = [self.Aimages + "{}".format(i) for i in os.listdir(self.Aimages)]
        all_plants = [self.Pimages + "{}".format(i) for i in
                                   os.listdir(self.Pimages)]

        # get number of samples
        num_samples_all_felis_catus = len(all_animals)
        num_samples_all_populus_trichocarpa = len(all_plants)

        # create labels
        populus_trichocarpa_labels = [0] * num_samples_all_populus_trichocarpa
        felis_catus_labels = [1] * num_samples_all_felis_catus

        # combine the datasets
        all_samples = all_animals + all_plants
        all_labels = felis_catus_labels + populus_trichocarpa_labels

        # shuffle the data and labels in the same way
        combined_samples_and_lists = list(zip(all_samples, all_labels))
        random.shuffle(combined_samples_and_lists)
        all_samples, all_labels = zip(*combined_samples_and_lists)

        self.split_rgb(all_animals, all_plants)
        return all_samples, all_labels

    def split_rgb(self, all_animals, all_plants):
        # split images into R, G, B floats
        resize_int = 50

        imgA = Image.open(all_animals[90])
        imgA = imgA.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
        animal_zero_as_floats = skimage.img_as_float(imgA)

        imgP = Image.open(all_plants[90])
        imgP = imgP.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
        plant_zero_as_floats = skimage.img_as_float(imgP)

        print(animal_zero_as_floats)
        print(plant_zero_as_floats)

        self.display_plots(imgA, animal_zero_as_floats)
        self.display_plots(imgP, plant_zero_as_floats)

    def display_plots(self, imX, X_im):
        display.display(imX)
        plt.figure()
        plt.imshow(X_im[:, :, 0], cmap="Reds_r")
        plt.colorbar()
        plt.grid(False)
        plt.show()

        display.display(imX)
        plt.figure()
        plt.imshow(X_im[:, :, 1], cmap="Greens_r")
        plt.colorbar()
        plt.grid(False)
        plt.show()

        display.display(imX)
        plt.figure()
        plt.imshow(X_im[:, :, 2], cmap="Blues_r")
        plt.colorbar()
        plt.grid(False)
        plt.show()

        display.display(imX)
        plt.figure()
        plt.imshow(X_im[:, :, :])
        plt.colorbar()
        plt.grid(False)
        plt.show()



