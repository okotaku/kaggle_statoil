# -*- coding: utf-8 -*-
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Concatenate
from keras.layers import Conv2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def DenseLike():
    inp = Input(shape=(75, 75, 3))

    x = Conv2D(16, kernel_size=(3, 3), activation='relu',
               input_shape=(75, 75, 3), padding='same')(inp)
    concat_list = [x]

    x = _block(x, 12)
    concat_list.append(x)
    x = Concatenate()(concat_list)

    x = _block(x, 12)
    concat_list.append(x)
    x = Concatenate()(concat_list)

    x = _block(x, 40)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    concat_list = [x]

    x = _block(x, 12)
    concat_list.append(x)
    x = Concatenate()(concat_list)

    x = _block(x, 12)
    concat_list.append(x)
    x = Concatenate()(concat_list)

    x = _block(x, 40)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model


def _block(x, c):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(c, kernel_size=(3, 3), activation='relu',
               padding='same')(x)
    x = Dropout(0.2)(x)

    return x
