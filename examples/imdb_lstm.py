#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.utils import np_utils
'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage 
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes: 

    - RNNs are tricky. Choice of batch size is important, 
    choice of loss and optimizer is critical, etc. 
    Most configurations won't converge.

    - LSTM loss decrease during training can be quite different 
    from what you see with CNNs/MLPs/etc. It's more or less a sigmoid
    instead of an inverse exponential.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    250s/epoch on GPU (GT 650M), vs. 400s/epoch on CPU (2.4Ghz Core i7).
'''

max_features=20000 # 字的频率,对sample里面的每一维特这个做截断,被截断说明出现频率太低
maxlen = 100 # 每个sample字的长短 cut texts after this number of words (among top max_features most common words)
batch_size = 16

print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# 如果是binary的label想用categorical的mode必须做一步转换
Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 256)) # embedding: 从 [nb_samples, maxlen] 到 [nb_samples, maxlen, max_features]
#model.add(LSTM(256, 128, return_sequences=True)) # 此时time_step已经被卷到output_dim里面去了[nb_samples, output_dim]
#model.add(Dropout(0.5))
model.add(LSTM(256, 32)) # 此时time_step已经被卷到output_dim里面去了[nb_samples, output_dim]
model.add(Dropout(0.5))
model.add(Dense(32, 2))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=0.01, epsilon=1e-6), class_mode="categorical")

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100, validation_split=0.25, show_accuracy=True)
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print('Test score:', score)

classes = model.predict_classes(X_test, batch_size=batch_size)
acc = np_utils.accuracy(classes, Y_test)
print('Test accuracy:', acc)

