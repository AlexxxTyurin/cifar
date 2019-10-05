import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


def identity_block(x, f, filters, stage, block):

    f1, f2, f3 = filters
    x_shortcut = x

    # The main path
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)(x)

    # The shortcut path
    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)

    return x


def convolutional_block(x, f, filters, stage, block, s=2):

    x_shortcut = x
    f1, f2, f3 = filters

    # The main path
    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer=glorot_uniform(0))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Shortcut path
    x_shortcut = Conv2D(f3, kernel_size=(1, 1), strides=(s, s), kernel_initializer=glorot_uniform(0), padding='valid')(x)
    x_shortcut = BatchNormalization(axis=3)(x_shortcut)

    x = Add()([x_shortcut, x])
    x = Activation('relu')(x)

    return x


def ResNet50(input_shape, classes):
    x_input = Input(input_shape)

    x = ZeroPadding2D(padding=(3, 3))(x_input)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = convolutional_block(x, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    x = convolutional_block(x, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    x = convolutional_block(x, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    x = convolutional_block(x, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    x = AveragePooling2D((2, 2), name='avg_pool')(x)

    ### END CODE HERE ###

    # output layer
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model





