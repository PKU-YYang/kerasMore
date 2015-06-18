# -*- coding:utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-15

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)

    # scale preds so that the class probas of each sample sum to 1
    # 全场最核心的一句话:y_pred.sum()做了归一化，终于完成了从softmax到概率的转换
    # ！！！！！！！！！！！！！！！！！！！！！！！
    # 对于y_pred, Each row represents a distribution
    y_pred /= y_pred.sum(axis=1, keepdims=True)
    # categorical_crossentopy就是把每个sample应该对应的label的概率加起来，因为y_true是one-hot序列，所以只有第一名有贡献
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error


# get_from_module是个内部函数，用来把str调用成真正的函数
# 在第二个参数里找第一个函数
from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

def to_categorical(y):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y
