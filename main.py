import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras.models import Model

def one_hot(matrix, classes):
    classes = tf.constant(classes, name='classes')
    one_hot_matrix = tf.one_hot(matrix, classes, name='one_hot', axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)

    return one_hot

# Putting all the training files into an array
folder = '/Users/alextyurin/Desktop/pycharm_projects/cifar/train'
im_files = os.listdir(folder)
EXTENSION = '.png'

data = np.array([plt.imread(folder + '/' + el) for el in im_files])
num_examples = data.shape[0]

X_train = data[0: int(num_examples * 0.8):, :, :, :]
X_train = np.reshape(X_train, [int(num_examples * 0.8), 32, 32, 3])

X_dev = data[int(num_examples * 0.8):, :, :, :]
X_dev = np.reshape(X_dev, [int(num_examples * 0.2), 32, 32, 3])

print('The shape of X_train: ', X_train.shape)
print('The shape of X_dev: ', X_dev.shape)

# def gestures_model(input_shape):
#     X_input = Input(input_shape)
#     X = ZeroPadding2D((2, 2))(X_input)
#
#     X = Conv2D(16, (3, 3), strides=(1, 1), name='conv1', activation='relu')(X)
#     X = BatchNormalization(axis=3, name='bn1')(X)
#     X = Conv2D(32, (3, 3), strides=(1, 1), name='conv2', activation='relu')(X)
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool1')(X)
#
#     X = Conv2D(64, (3, 3), strides=(1, 1), name='conv3', activation='relu')(X)
#     X = BatchNormalization(axis=3, name='bn2')(X)
#     X = Conv2D(128, (3, 3), strides=(1, 1), name='conv4', activation='relu')(X)
#     X = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool2')(X)
#
#     X = Conv2D(120, (3, 3), strides=(1, 1), name='conv5', activation='relu')(X)
#     X = AveragePooling2D((2, 2), strides=(1, 1), name='avg_pool1')(X)
#
#     X = Flatten()(X)
#     X = Dense(26, activation='softmax', name='fc3')(X)
#
#     model = Model(inputs=X_input, outputs=X, name='gestures_model')
#
#     return model
#
#
# model = gestures_model(X_train.shape[1:])
# model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs=100, batch_size=1000)
#
# print(model.evaluate(X_train, Y_train))
#
# predictions = model.evaluate(X_dev, Y_dev)
#
# print("Loss = " + str(predictions[0]))
# print("Dev Accuracy = " + str(predictions[1]))
# print("Prediction", model.predict(X_dev))
# print("True results", Y_dev)
