# -*- coding: utf-8 -*-
from keras.layers import concatenate, Dense, Dropout, Input, MaxPooling2D, Activation
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def ReslikeAnglenet():
    input_conv = Input(shape=[75, 75, 3])

    skip = Conv2D(32, kernel_size=(3, 3), activation='relu',
                  input_shape=(75, 75, 3), padding='same')(input_conv)
    x = _residual(input_conv, 32)
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
    x = GlobalMaxPooling2D()(x)

    input_ang = Input(shape=[1])

    x = concatenate([x, input_ang])
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input=[input_conv, input_ang], output=x)

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
