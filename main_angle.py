#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(7)

from make_angledf import make_df
from model.vgg_anglenet import VggAngleNet
from model.small_anglenet import SmallAnglenet
from model.reslike_anglenet import ReslikeAnglenet


# set gpu usage
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="1",
                                                  allow_growth=True))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


def gen_flow_for_two_inputs(X1, X2, y, batch_size):
    genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=7)
    genX2 = gen.flow(X1, X2, batch_size=batch_size, seed=7)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X2i[1][np.isnan(X2i[1])] = 0
            yield [X1i[0], X2i[1]], X1i[1]


if __name__ == "__main__":
    x, y, x_ang = make_df("./data/train.json", "train")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=7).split(x, y))
    result = []
    for j, (train_idx, test_idx) in enumerate(folds):
        xtr = x[train_idx]
        xtr_ang = x_ang[train_idx]
        ytr = y[train_idx]
        xval = x[test_idx]
        xval_ang = x_ang[test_idx]
        yval = y[test_idx]
        #model = VggAngleNet()
        #model = SmallAnglenet()
        model = ReslikeAnglenet()
        optimizer = Adam(lr=0.0001, decay=0.0)
        #optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=5,
                                      verbose=0, mode='min')
        ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.18,
                                           patience=5, verbose=2, epsilon=1e-4,
                                           mode='min')

        gen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 width_shift_range=0,
                                 height_shift_range=0,
                                 channel_shift_range=0,
                                 zoom_range=0.2,
                                 rotation_range=10)
        gen_flow = gen_flow_for_two_inputs(xtr, xtr_ang, ytr, batch_size=64)
        model.fit_generator(gen_flow, steps_per_epoch=len(xtr), epochs=50,
                            callbacks=[earlyStopping, ckpt],
                            verbose=2,
                            validation_data=([xval, xval_ang], yval))

        model.load_weights(filepath='.model.hdf5')
        score = model.evaluate([xval, xval_ang], yval, verbose=1)
        print('Test score:', score[0], 'Test accuracy:', score[1])

        xtest, df_test, xtest_ang = make_df("./data/test.json", "test")
        pred_test = model.predict([xtest, xtest_ang]).reshape(-1)
        result.append(pred_test)
    pd.DataFrame(result).to_csv('result.csv', index=False)
