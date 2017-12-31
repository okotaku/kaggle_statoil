# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dropout, Flatten, Dense
from keras.layers.core import Activation
from keras.models import Sequential, Model


def Vgg16(freeze_leyer):
    input_tensor = Input(shape=(75, 75, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256))
    top_model.add(Activation('relu'))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(input=vgg16.input, output=top_model(vgg16.output))
    if freeze_leyer > 0:
        for layer in model.layers[:freeze_leyer]:
            layer.trainable = False

    return model
