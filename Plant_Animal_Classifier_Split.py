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
import numpy as np
import matplotlib.pyplot as plt


class Plant_Animal_Classifier:
    def __init__(self, classnames, dir_1, dir_2, resize_int=50):
        self.classnames = classnames
        self.dir_1_list = False
        self.dir_2_list = False
        self.resize_int = resize_int

        self.Pimages = dir_1
        self.Aimages = dir_2       
        
        if type(dir_1) is list:
            self.dir_1_list = True

        if type(dir_2) is list:
            self.dir_2_list = True
        
        self.model = None
        self.pca_R = None
        self.pca_G = None
        self.pca_B = None
    
    def save_to_file_test(self, filename):
        # np.save(filename, self.model)
        self.model.save(filename)
    
    
    def main_loop(self):

        all_samples, all_labels = self.split_categorically()

        all_images, all_images_r, all_images_g, all_images_b =\
            self.create_lists(all_samples, self.resize_int)

        all_pca_r , all_pca_g, all_pca_b =\
            self.pca_alg(all_images, self.resize_int, all_images_r, all_images_g, all_images_b)

        X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B, test_labels, train_labels = \
            self.split_train_rgb(all_pca_r, all_pca_g, all_pca_b, all_labels)

        X_train, X_test = self.create_test_train_lists(X_train_R, X_train_G, X_train_B, X_test_R, X_test_G, X_test_B)

        test_loss, test_acc = self.train_compile_model(X_train, X_test, train_labels, test_labels)

        self.get_results(test_loss, test_acc)

    def get_results(self, test_lost, test_acc):
        print('\nTest accuracy:', test_acc)

    def train_compile_model(self, X_train, X_test, train_labels, test_labels):
        # train model
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model.fit(np.array(X_train), np.array(train_labels), epochs=10)

        test_loss, test_acc = self.model.evaluate(np.array(X_test), np.array(test_labels), verbose=2)

        return test_loss, test_acc
    
    
    def return_classifier(self):
        return self.model

    
    def predict_using_trained_model(self, images_dir, plot=False): #, typepassedin1, typepassedin2):
        all_images_directory = [images_dir + "{}".format(i) for i in os.listdir(images_dir)]
                
        _, vals_R, vals_G, vals_B = self.create_lists(all_images_directory, self.resize_int)
        
        images_flattened_R = []
        for image in vals_R:
            images_flattened_R.append(np.array(image).flatten())
 
 
        images_flattened_G = []
        for image in vals_G:
            images_flattened_G.append(np.array(image).flatten())
 
        images_flattened_B = []
        for image in vals_B:
            images_flattened_B.append(np.array(image).flatten())
             
        images_principal_components_R = self.pca_R.transform(images_flattened_R)
        images_principal_components_G = self.pca_G.transform(images_flattened_G)
        images_principal_components_B = self.pca_B.transform(images_flattened_B)

        # print(images_principal_components_R)
        # images_principal_components_R_tans = self.pca_R.fit_transform(images_flattened_R)
        # print(images_principal_components_R_tans)
        X = []
        for i in range(len(images_principal_components_R)):
            X.append(np.concatenate((images_principal_components_R[i], images_principal_components_G[i], images_principal_components_B[i])))

        predictions = self.model.predict_classes(np.array(X))
        
        predictions_to_return = []
        for p in predictions:
            if p == 0:
               predictions_to_return.append(self.classnames[0])
            else:
                predictions_to_return.append(self.classnames[1])
                
        if plot:
            for i in range(len(predictions_to_return)):
                print(predictions_to_return[i])
                display.display(Image.open(all_images_directory[i]))
        
        return predictions_to_return
        # predictions.replace(0, typepassedin1)
        # predictions.replace(1, typepassedin2)

   #     predictions = np.where(predictions == 0, typepassedin1, predictions)
    #    predictions = np.where(predictions == 1, typepassedin2, predictions)

#        count_zero = 0
#        count_one = 0
#        count_total = 0
#
#        for i in predictions:
#            if i == typepassedin1:
#                count_zero += 1
#                count_total += 1
#            elif i == typepassedin2:
#                count_one += 1
#                count_total += 1
#            else:
#                count_total += 1
#
#        accuracy = count_zero/count_total
     #   return predictions, accuracy, typepassedin1, typepassedin2
        

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

        self.pca_R = PCA(n_components=number_of_components)
        self.pca_G = PCA(n_components=number_of_components)
        self.pca_B = PCA(n_components=number_of_components)

        all_images_principal_components_R = self.pca_R.fit_transform(all_images_flattened_R)
        all_images_principal_components_G = self.pca_G.fit_transform(all_images_flattened_G)
        all_images_principal_components_B = self.pca_B.fit_transform(all_images_flattened_B)

        return all_images_principal_components_R, all_images_principal_components_G, all_images_principal_components_B

    def create_lists(self, all_samples, resize_int):
        all_images = []
        for sample in all_samples:
            all_images.append(self.reshape_image_and_convert_to_float(sample, resize_int))
        all_images_R = []
        all_images_G = []
        all_images_B = []
        
        for image in all_images:
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
        if self.dir_1_list == False:
            all_animals = [self.Aimages + "{}".format(i) for i in os.listdir(self.Aimages)]
        else:
            all_animals = []
            for directory in self.Aimages:
                temp = [directory + "{}".format(i) for i in os.listdir(directory)]
                for t in temp:
                    all_animals.append(t)
        
        if self.dir_2_list == False:
            all_plants = [self.Pimages + "{}".format(i) for i in os.listdir(self.Pimages)]
        else:
            all_plants = []
            for directory in self.Pimages:
                temp = [directory + "{}".format(i) for i in os.listdir(directory)]
                for t in temp:
                    all_plants.append(t)

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

   #     self.split_rgb(all_animals, all_plants)
        return all_samples, all_labels

    def split_rgb(self, all_animals, all_plants):
        # split images into R, G, B floats
        resize_int = 50
        # randomly choose a picture from given directory to show
        imgAInt = random.randint(0, len(all_animals))
        imgBInt = random.randint(0, len(all_plants))

        imgA = Image.open(all_animals[imgAInt])
        imgA = imgA.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
        animal_zero_as_floats = skimage.img_as_float(imgA)

        imgP = Image.open(all_plants[imgBInt])
        imgP = imgP.resize((resize_int, resize_int), PIL.Image.ANTIALIAS)
        plant_zero_as_floats = skimage.img_as_float(imgP)

        # print(animal_zero_as_floats)
        # print(plant_zero_as_floats)

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



