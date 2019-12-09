import cv2
import numpy as np
import os
import tflearn
import shutil
import math
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import sklearn
from sklearn.metrics import accuracy_score
import tensorflow as tf



class CNN_Classification:
    def __init__(self, classnames, dir1, dir2, resize_int=48):
        self.classnames = classnames
        self.class1images = dir1
        self.class2images = dir2
        self.IMG_SIZE = resize_int
        self.dir_1_list = False
        self.dir_2_list = False
        self.model = None
        #
        self.TRAIN_DIR = "C:\\Users\\samic\\Documents\\ML-final-project\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TRAIN"
        self.TEST_DIR = "C:\\Users\\samic\\Documents\\ML-final-project\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TEST-"+str(self.classnames[0]+"\\")

      #  self.TRAIN_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TRAIN\\"
      #  self.TEST_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TEST-"+str(self.classnames[0]+"\\")

        if type(dir1) is list:
            self.dir_1_list = True

        if type(dir2) is list:
            self.dir_2_list = True


    def main_loop(self):
        self.move_to_folders()

        LR = 1e-6
        MODEL_NAME = 'CNN-{}-vs-{}.model'.format(self.classnames[0], self.classnames[1])
        train_data = self.use_train_data()
        model = self.neural_net_architecture(LR)
        # #print(model)
        self.model = model

        train = train_data[:-20]
        test = train_data[-20:]
        X = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        test_y = [i[1] for i in test]
        self.model = model

        # Train the network
#        self.model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
#                       show_metric=True, run_id=MODEL_NAME)
        self.model.fit({'input': X}, {'targets': Y}, n_epoch=10, show_metric=True, run_id=MODEL_NAME)
        
        predicted_y_temp = self.model.predict(test_x)
        predicted_y = np.argmax(predicted_y_temp, axis = 1)
        actual_y = np.argmax(test_y, axis = 1)
        test_accuracy = accuracy_score(predicted_y, actual_y)

        print("Test accuracy: ", test_accuracy)
        # tensorboard --logdir="C:\Users\djenz\OneDrive - University of Vermont\Machine-Learning-Final-Project\CNN TUT\log"
        self.model.save(MODEL_NAME)

        self.plot_figures()



    def move_to_folders(self):
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(self.TRAIN_DIR):
            for file in f:
                files.append(os.path.join(r, file))
        for i in range(len(files)):
            os.remove(str(files[i]))

        classnames = self.classnames
        src_class1 = self.class1images
        src_class2 = self.class2images


        dest_TEST = self.TEST_DIR
        dest_TRAIN = self.TRAIN_DIR

        lenfolder1 = len([name for name in os.listdir(src_class1) if os.path.isfile(os.path.join(src_class1, name))])
        lenfolder2 = len([name for name in os.listdir(src_class2) if os.path.isfile(os.path.join(src_class2, name))])

        folder1 = [src_class1 + "{}".format(i) for i in os.listdir(src_class1)]
        folder2 = [src_class2 + "{}".format(i) for i in os.listdir(src_class2)]

        trainnum1 = math.floor(lenfolder1*.8)
        testnum1 = lenfolder1 - trainnum1
        trainnum2 = math.floor(lenfolder2*.8)
        testnum2 = lenfolder2 - trainnum2

        # place 80% of images into dest_train
        for i in range(int(trainnum1)):
            src = folder1[i]
            shutil.copy(src, dest_TRAIN)

        for i in range(int(trainnum2)):
            src = folder2[i]
            shutil.copy(src, dest_TRAIN)

        # place remaining 20% of images into dest_test
        for i in range(int(trainnum1),int(testnum1)):
            src = folder1[i]
            shutil.copy(src, dest_TEST)

        for i in range(int(trainnum2), int(testnum2)):
            src = folder2[i]
            shutil.copy(src, dest_TEST)




    def plot_figures(self):
        # if you don't have this file yet
        test_data = self.process_test_data()
        # if you already have it
        #test_data = np.load('test_data.npy', allow_pickle=True)

        fig = plt.figure()

        for num, data in enumerate(test_data[:25]):
            # cat : [1,0]
            # dog : [0,1]

      #      img_num = data[1]
            img_data = data[0]
            y = fig.add_subplot(5, 5, num + 1)
            orig = img_data
            data_reshaped = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

            model_out = self.model.predict([data_reshaped])[0]
            
            if np.argmax(model_out) == 0:
                str_label = self.classnames[0][:5]
            else:
                str_label = self.classnames[1][:5]

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()


    def label_img(self, img):
        word_label = img.split('_')[-3]
        if self.classnames[0] == "lycophyte":
            classnames_in = ["arabidopsis", "carica", "medicago", "populus", "vitis", "oryza", "sorghum"]
        if self.classnames[0] == "monocot":
            classnames_in = ["arabidopsis", "carica", "medicago", "populus", "vitis"]
        if self.classnames[0] == "oryza":
            classnames_in = ["sorghum"]
        if self.classnames[0] == "dicot1":
            classnames_in = ["populus", "medicago"]
        else:
            classnames_in = [word_label]
            
        for i in classnames_in:
            # print(word_label)
            if word_label == str(i):
                return [1,0]

        return [0,1]

    def create_train_data(self):
        training_data = []
        for img in tqdm(os.listdir(self.TRAIN_DIR)):
            label = self.label_img(img)
            path = os.path.join(self.TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.IMG_SIZE, self.IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        # print(label)
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def process_test_data(self):
        testing_data = []
        for img in tqdm(os.listdir(self.TEST_DIR)):
            path = os.path.join(self.TEST_DIR, img)
            img_num = img.split('_')[-3]
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.IMG_SIZE, self.IMG_SIZE))
            testing_data.append([np.array(img), img_num])

        np.save('test_data.npy', testing_data)
        return testing_data

    def use_train_data(self):
        # if we dont have trained data yet
        train_data = self.create_train_data()
        # if already have trained data
        # train_data = np.load('train_data.npy')
        return train_data

    def neural_net_architecture(self, LR):
        tf.reset_default_graph()

        convnet = input_data(shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1], name='input')

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)


        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', batch_size=8, learning_rate=LR, loss='binary_crossentropy', name='targets')
        #tensorboard --logdir="C:\Users\djenz\OneDrive - University of Vermont\Machine-Learning-Final-Project\log"
        model = tflearn.DNN(convnet, tensorboard_dir='log')

        return model
















