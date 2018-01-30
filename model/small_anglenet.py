# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.layers import concatenate, Dense, Dropout, Input, MaxPooling2D, Flatten
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def SmallAnglenet():
    input_conv = Input(shape=[75, 75, 3])
    x = Conv2D(64, kernel_size=(3, 3), activation='relu',
               input_shape=(75, 75, 3))(input_conv)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    #x = Flatten()(x)
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
