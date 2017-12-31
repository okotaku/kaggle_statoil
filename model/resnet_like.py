# -*- coding: utf-8 -*-
from keras.layers import Activation, Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def ResnetLike():
    inp = Input(shape=(75, 75, 3))

    skip = Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(75, 75, 3), padding='same')(inp)
    x = _residual(inp, 32)
    x = add([skip, x])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    skip = Conv2D(64, kernel_size=(3, 3), activation='relu',
                  padding='same')(x)
    x = _residual(x, 64)
    x = add([skip, x])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    skip = Conv2D(128, kernel_size=(3, 3), activation='relu',
                  padding='same')(x)
    x = _residual(x, 128)
    x = add([skip, x])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    skip = Conv2D(256, kernel_size=(3, 3), activation='relu',
                  padding='same')(x)
    x = _residual(x, 256)
    x = add([skip, x])
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def _residual(x, c):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(c, kernel_size=(3, 3), activation='relu',
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(c, kernel_size=(3, 3), activation='relu',
               padding='same')(x)

    return x
