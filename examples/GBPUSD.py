#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1, JZS2, JZS3
from keras.datasets import imdb
from keras.utils import np_utils
from sklearn import preprocessing
import cPickle,sys
import copy

def load_data(filename, test_split=0.3, n_timestep=8, n_features=19, normalisation=False):

    raw = np.loadtxt(filename,delimiter=',', dtype=float,skiprows=1)
    labels = raw[:,0] # first column is the label
    X = raw[:,1:]

    random.seed(89757)
    random.shuffle(X)
    random.seed(89757)
    random.shuffle(labels)


    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    # Convert class vector (integers from 0 to nb_classes)
    # to binary class matrix, for use with categorical_crossentropy
    Y_train = np_utils.to_categorical(y_train, len(np.unique(y_train)))
    Y_test = np_utils.to_categorical(y_test, len(np.unique(y_train)))

    if normalisation:

        scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # change to 3D tensor
    assert X_train.shape[1] == n_timestep*n_features, "dimension not consistent"

    _train = X_train[:,:,np.newaxis]
    X_train = _train.reshape(X_train.shape[0],n_features,n_timestep).transpose(0,2,1)
    _train = X_test[:,:,np.newaxis]
    X_test = _train.reshape(X_test.shape[0],n_features,n_timestep).transpose(0,2,1)


    # xtrain: [n_samples, n_timesteps, n_features]
    return (X_train, Y_train), (X_test, Y_test)


def build_model(X_train, Y_train, X_test, Y_test, batch_size=16, n_epoch=200, hidden=None):

    model = Sequential()

    for i in hidden:
        model.add(JZS3(19, i)) # 此时time_step已经被卷到output_dim里面去了[nb_samples, output_dim]
        model.add(Dropout(0.5))

    model.add(Dense(hidden[-1], 3))
    model.add(Activation('softmax'))

    optimiser = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    #optimiser = Adagrad(lr=0.01, epsilon=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, class_mode="categorical")

    print("Train...")

    best_acc = 0.
    best_model = model
    for i in range(n_epoch):

        print("Epoch:%d ..." % i)
        performance = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, validation_split=0.15,
                                show_accuracy=True, verbose=0, shuffle=False)
        print("loss on training set is %.2f, accuracy is %.2f %%" % (performance.loss[0]*100, performance.accuracy[0]*100))
        # early stopping
        if performance.validation_accuracy > best_acc:
            best_acc = performance.validation_accuracy
            best_model = copy.deepcopy(model)  # 必须用深拷贝，否则模型不变
            print("up-till-now best accuracy on validation set is %.2f %%" % (best_acc[0]*100))

    score = best_model.evaluate(X_test, Y_test, batch_size=1, show_accuracy=True)

    sys.setrecursionlimit(2000) # if not increase this, the model is too complicated to save
    cPickle.dump(best_model,open("./GBPUSD.pkl","wb"))

    print('Test accuracy %.2f %%:' % (score[1]*100))


if __name__ == "__main__":

    (X_train, Y_train), (X_test, Y_test) = load_data(filename='example_data/GBPUSD.csv', test_split=0.3,
                                                     n_timestep=8, n_features=19, normalisation=False)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    build_model(X_train, Y_train, X_test, Y_test, batch_size=4, n_epoch=150, hidden=[64])
