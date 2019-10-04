import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
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
    x = Conv2D(filters=f1, )



