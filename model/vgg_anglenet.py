# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.layers import concatenate, Dense, Dropout, Input, MaxPooling2D, Flatten
from keras.layers import Conv2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


def VggAngleNet():
    vgg16 = VGG16(weights='imagenet', include_top=False,
                  input_shape=(75, 75, 3))
    for layer in vgg16.layers:
        layer.trainable = False
    x1 = GlobalMaxPooling2D()(vgg16.output)

    mbnet = MobileNet(weights=None, alpha=0.9,
                      input_tensor=vgg16.input, include_top=False,
                      input_shape=(75, 75, 3))
    x2 = GlobalAveragePooling2D()(mbnet.output)

    input_ang = Input(shape=[1])

    x = concatenate([x1, x2, input_ang])
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input=[vgg16.input, input_ang], output=x)

    return model
