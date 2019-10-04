import keras
import tensorflow as tf
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.layers import Input, ZeroPadding2D, BatchNormalization, Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras.models import Sequential


def lenet_5():
    model = Sequential()

    model.add(ZeroPadding2D((2, 2)))

    model.add(Conv2D(6, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(AveragePooling2D())

    model.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization(axis=3))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(120, activation='relu'))

    model.add(Dense(84, activation='relu'))

    model.add(Dense(100, activation='softmax'))

    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
n, f_h, f_w, n_colors = x_train.shape

# Dividing data into dev and training sets
x_dev = x_train[int(0.8 * n):, :, :, :]
y_dev = y_train[int(0.8 * n):]

x_train = x_train[:int(n * 0.8), :, :, :]
y_train = y_train[:int(n * 0.8)]

print(x_train.shape)
print(y_train.shape)


# def one_hot(matrix, classes):
#     n = matrix.shape[0]
#     classes = tf.constant(classes, name='classes')
#     one_hot_matrix = tf.reshape(tf.one_hot(matrix, classes, name='one_hot', axis=0), (n, 10))
#     with tf.Session() as sess:
#         one_hot = sess.run(one_hot_matrix)
#
#     return one_hot


y_train = keras.utils.to_categorical(y_train, 10)
y_dev = keras.utils.to_categorical(y_dev, 10)


model = lenet_5()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=1000)

print(model.evaluate(x_dev, y_dev))
# predictions = model.evaluate(x_dev, y_dev)

# print(x_train[0].shape)
# plt.imshow(x_train[0])
# plt.show()