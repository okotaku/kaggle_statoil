# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.layers.core import Activation
from keras.models import Model


def Vgg16(freeze_leyer):
    input_tensor = Input(shape=(75, 75, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    x = Flatten()(vgg16.output)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input=vgg16.input, output=x)
    if freeze_leyer > 0:
        for layer in model.layers[:freeze_leyer]:
            layer.trainable = False

    return model
