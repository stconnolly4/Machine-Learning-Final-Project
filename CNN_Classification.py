import cv2
import numpy as np
import os
import tflearn
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
        self.resize_int = resize_int
        self.dir_1_list = False
        self.dir_2_list = False

        self.TRAIN_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TRAIN\\"
        self.TEST_DIR = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\CNN_TRAIN_TEST\\TEST\\"

        if type(dir1) is list:
            self.dir_1_list = True

        if type(dir2) is list:
            self.dir_2_list = True


    def main_loop():
        LR = 1e-6
        MODEL_NAME = 'CNN-{}-{}.model'.format(LR, '6conv')
        train_data = use_train_data()
        model = neural_net_architecture(LR)


        train = train_data[:-500]
        test = train_data[-500:]
        X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        test_y = [i[1] for i in test]

        # Train the network
        model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
        # tensorboard --logdir="C:\Users\djenz\OneDrive - University of Vermont\Machine-Learning-Final-Project\CNN TUT\log"
        model.save(MODEL_NAME)

        plot_figures()

        testing_data = process_test_data()


    def plot_figures():
        # if you don't have this file yet
        test_data = process_test_data()
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
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

            model_out = model.predict([data])[0]

            if np.argmax(model_out) == 1:
                str_label = self.classnames[0]
            else:
                str_label = self.classnames[1]

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()


    def label_img(img):
        word_label = img.split('.')[-3]
        if word_label == str(self.classnames[0]) : return [1,0]
        elif word_label == str(self.classnames[1]) : return [0,1]

    def create_train_data():
        train_data = []
        for img in tqdm(os.listdir(self.TRAIN_DIR)):
            label = label_img(img)
            path = os.path.join(self.TRAIN_DIR, img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        shuffle(training_data)
        np.save('train_data.npy', training_data)
        return training_data

    def process_test_data():
        testing_data = []
        for img in tqdm(os.listdir(self.TEST_DIR)):
            path = os.path.join(self.TEST_DIR, img)
            img_num = img.split('.')[0]
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), img_num])

        np.save('test_data.npy', testing_data)
        return testing_data

    def use_train_data():
        # if we dont have trained data yet
        train_data = create_train_data()
        # if already have trained data
        # train_data = np.load('train_data.npy')
        return train_data

    def neural_net_architecture(LR):
        tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

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

















