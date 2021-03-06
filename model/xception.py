# -*- coding: utf-8 -*-
from keras.applications.xception import Xception
from keras.layers import Input, Dropout, Flatten, Dense
from keras.layers.core import Activation
from keras.models import Sequential, Model


def Xception():
    input_tensor = Input(shape=(75, 75, 3))
    xc = Xception(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=xc.output_shape[1:]))
    top_model.add(Dense(512))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128))
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.1))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(input=xc.input, output=top_model(xc.output))
    for layer in xc.layers:
        layer.trainable = False

    return model
