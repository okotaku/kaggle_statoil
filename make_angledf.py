# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def make_df(path, mode):
    """
    params
    --------
    path(str): path to json
    mode(str): "train" or "test"

    outputs
    --------
    X(np.array): 画像の配列 shape=(None, 75, 75, 3)
    Y(np.array): ラベルの配列 shape=(None,)
    df(pd.DataFrame): テストデータのデータフレーム
    """
    minang = 24.7546
    maxang = 45.9375
    df = pd.read_json(path)
    df.inc_angle = df.inc_angle.replace('na', 0)
    df.inc_angle = df.inc_angle.fillna(0)
    X = _get_scaled_imgs(df)
    if mode == "test":
        return X, df, (df.inc_angle.values - minang) / (maxang - minang)

    Y = np.array(df['is_iceberg'])

    idx_tr = np.where(df.inc_angle > 0)

    X = X[idx_tr[0]]
    Y = Y[idx_tr[0], ...]
    X_ang = df.inc_angle[idx_tr[0]]

    return X, Y, (X_ang - minang) / (maxang - minang)


def _get_scaled_imgs(df):
    imgs = []

    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)
