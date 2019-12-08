import cv2
import numpy as np
import os
import tflearn
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

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

        self.TRAIN_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TRAIN\\"
        self.TEST_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TEST\\"

        if type(dir1) is list:
            self.dir_1_list = True

        if type(dir2) is list:
            self.dir_2_list = True


    def main_loop(self):
        LR = 1e-6
        MODEL_NAME = 'CNN-{}-{}.model'.format(LR, '6conv')
        train_data = self.use_train_data()
        model = self.neural_net_architecture(LR)
        self.model = model


        train = train_data[:-500]
        test = train_data[-500:]
        X = np.array([i[0] for i in train]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        test_y = [i[1] for i in test]

        # Train the network
        self.model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
        # tensorboard --logdir="C:\Users\djenz\OneDrive - University of Vermont\Machine-Learning-Final-Project\CNN TUT\log"
        self.model.save(MODEL_NAME)

        self.plot_figures()

        testing_data = self.process_test_data()

    def move_to_folders(self):
        class_chicken = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\gallina\\"
        class_dog = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\cane\\"
        class_kitty = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\gatto\\"
        class_elephant = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\elefante\\"
        class_horse = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\cavallo\\"
        class_sheep = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\pecora\\"
        class_squirrell = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\scoitattolo\\"
        class_mrmrscow = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\mucca\\"

        classnames = ["chicken", "dog", "cat", "elephant", "horse", "sheep", "squirrell", "cow"]
        sources = [class_chicken, class_dog, class_kitty, class_elephant, class_horse, class_sheep, class_squirrell, class_mrmrscow]

        for i in range(len(8)):
            if self.classnames[0] == classnames[i]:
                src_class1 = sources[i]
            if self.classnames[1] == classnames[i]:
                src_class2 = sources[i]


        dest_TEST = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TEST\\"
        dest_TRAIN = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TRAIN\\"

        src_class1 = ""
        src_class2 = ""


        src_TEST = []
        SRC_TRAIN = []



    def plot_figures(self):
        # if you don't have this file yet
        test_data = self.process_test_data()
        # if you already have it
        #test_data = np.load('test_data.npy', allow_pickle=True)

        fig = plt.figure()

        for num, data in enumerate(test_data[:12]):
            # cat : [1,0]
            # dog : [0,1]

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)

            model_out = self.model.predict([data])[0]

            if np.argmax(model_out) == 1:
                str_label = self.classnames[0]
            else:
                str_label = self.classnames[1]

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()


    def label_img(self, img):
        word_label = img.split('.')[-3]
        if word_label == str(self.classnames[0]) : return [1,0]
        elif word_label == str(self.classnames[1]) : return [0,1]

    def create_train_data(self):
        training_data = []
        for img in tqdm(os.listdir(self.TRAIN_DIR)):
            label = self.label_img(img)
            path = os.path.join(self.TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (self.IMG_SIZE, self.IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def process_test_data(self):
        testing_data = []
        for img in tqdm(os.listdir(self.TEST_DIR)):
            path = os.path.join(self.TEST_DIR, img)
            img_num = img.split('.')[0]
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

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 32, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = conv_2d(convnet, 64, 2, activation='relu')
        convnet = max_pool_2d(convnet, 2)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        return model

















