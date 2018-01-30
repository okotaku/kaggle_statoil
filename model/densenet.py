from keras_contrib.applications.densenet import DenseNet
from keras.layers import Dense, Flatten
from keras.models import Model


def Densenet():
    densenet = DenseNet(input_shape=(75, 75, 3))
    x = Flatten()(densenet.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(input=densenet.input, output=x)

    return model
