#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(7)

from make_df import make_df
from model.starter_model import StarterModel
from model.vgg16 import Vgg16


if __name__ == "__main__":
    x, y = make_df("./data/train.json", "train")
    xtr, xval, ytr, yval = train_test_split(x, y, test_size=0.25,
                                            random_state=7)
    #model = StarterModel()
    model = Vgg16()
    optimizer = Adam(lr=0.001, decay=0.0)
    #optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0,
                                  mode='min')
    ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,
                           monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=7, verbose=1, epsilon=1e-4,
                                       mode='min')

    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0,
                             height_shift_range=0,
                             channel_shift_range=0,
                             zoom_range=0.2,
                             rotation_range=10)
    gen.fit(xtr)
    model.fit_generator(gen.flow(xtr, ytr, batch_size=32),
                        steps_per_epoch=len(xtr), epochs=50,
                        callbacks=[earlyStopping, ckpt, reduce_lr_loss],
                        validation_data=(xval, yval))

    model.load_weights(filepath='.model.hdf5')
    score = model.evaluate(xtr, ytr, verbose=1)
    print('Train score:', score[0], 'Train accuracy:', score[1])

    xtest, df_test = make_df("./data/test.json", "test")
    pred_test = model.predict(xtest)
    print("ok pred")
    pred_test = pred_test.reshape((pred_test.shape[0]))
    print("reshape")
    submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test})
    submission.to_csv('submission.csv', index=False)
