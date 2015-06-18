#-*-: coding:utf-8 -*-
from __future__ import absolute_import
import numpy
def load_data(trainset='lcg/DL_train.csv',validset='lcg/DL_valid.csv',testset='lcg/DL_test.csv'):

    #分别读入三个文件并share他们

    data=numpy.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set_x, valid_set_y =(data[:,:-1],data[:,-1])

    data=numpy.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set_x, test_set_y =(data[:,:-1],data[:,-1])

    data=numpy.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set_x, train_set_y=(data[:,:-1],data[:,-1])

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval