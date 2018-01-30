#!/usr/bin/env python
# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(7)

from make_df import make_df
from model.densenet_like import DenseLike
from model.resnet_like import ResnetLike
from model.small_cnn import SmallCNN
from model.vgg_like import VggLike
from model.vgg16 import Vgg16
from model.xception import Xception
from model.jirkamodel import JirkaModel


# set gpu usage
config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list="0",
                                                  allow_growth=True))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

if __name__ == "__main__":
    x, y = make_df("./data/train.json", "train")
    folds = list(StratifiedKFold(n_splits=10, shuffle=True,
                                 random_state=7).split(x, y))
    result = []
    for j, (train_idx, test_idx) in enumerate(folds):
        xtr = x[train_idx]
        ytr = y[train_idx]
        xval = x[test_idx]
        yval = y[test_idx]
        #model = DenseLike()
        #model = ResnetLike()
        #model = SmallCNN()
        #model = VggLike()
        #model = Vgg16(freeze_leyer=15)
        #model = SimpleNet()
        model = JirkaModel()
        optimizer = Adam(lr=0.0001, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=5,
                                      verbose=0, mode='min')
        ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,
                               monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.18,
                                           patience=5, verbose=1, epsilon=1e-4,
                                           mode='min')

        gen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 channel_shift_range=0,
                                 zoom_range=0.2,
                                 rotation_range=10)
        gen.fit(xtr)
        model.fit_generator(gen.flow(xtr, ytr, batch_size=64),
                            steps_per_epoch=len(xtr), epochs=50,
                            callbacks=[earlyStopping, ckpt],
                            validation_data=(xval, yval))

        model.load_weights(filepath='.model.hdf5')
        score = model.evaluate(xval, yval, verbose=1)
        print('Test score:', score[0], 'Test accuracy:', score[1])

        xtest, df_test = make_df("./data/test.json", "test")
        pred_test = model.predict(xtest).reshape(-1)
        result.append(pred_test)
    pd.DataFrame(result).to_csv('result.csv', index=False)
