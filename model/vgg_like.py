# -*- coding: utf-8 -*-
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


def VggLike():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     input_shape=(75, 75, 3), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    return model
