from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
import lcg_data
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop,Adadelta
from keras.utils import np_utils
import numpy as np

'''
    Train a simple deep NN on the MNIST dataset.
'''

batch_size = 50
nb_classes = 2
nb_epoch = 200

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, Y_train),(X_valid, Y_valid) ,(X_test, Y_test) = lcg_data.load_data()


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_valid = X_valid.astype("float32")
print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)
Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
# convert class vectors to binary class matrices


model = Sequential()
model.add(Dense(30, 10))
model.add(Activation('sigmoid'))
# model.add(Dropout(0.5))
# model.add(Dense(128, 1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1024, 1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1024, 128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, 2))
model.add(Activation('softmax'))

#rms = RMSprop()
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adadelta)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_valid, Y_valid))
score = model.evaluate(X_valid, Y_valid,batch_size=X_valid.shape[0], show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('predict',model.predict(X_test))
print('predict class',model.predict_classes(X_test))