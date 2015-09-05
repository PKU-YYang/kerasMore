#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar, printv
import numpy as np
from scipy.stats import itemfreq
import theano.tensor as T



def load_data(trainset, validset, testset):

    # 分别读入三个文件并share他们
    data = np.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set=(data[:,:-2],data[:,-2],data[:,-1])

    data = np.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set=(data[:,:-2],data[:,-2],data[:,-1]) #feature,label,raceid

    data = np.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set=(data[:,:-2],data[:,-2],data[:,-1])

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y, data_index = data_xy

        # index实际的样子是从0开始每一位记录每组赛事的第一名的位置，也就是每组比赛开始的地方，最后一位是全部输入sample的行数
        data_index = np.concatenate((np.array([0]), np.cumsum(itemfreq(data_index)[:,1])))

        return data_x.astype('float32'), data_y.astype('int32'), data_index.astype('int32')

    # 主要是用x和index，y未来再使用
    test_set_x, test_set_y, test_set_index = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_index = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_index = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y, train_set_index), (valid_set_x, valid_set_y, valid_set_index),
            (test_set_x, test_set_y, test_set_index)]
    return rval

dataset = ['./example_data/horse_train.csv','./example_data/horse_valid.csv','./example_data/horse_test.csv']
datasets = load_data(*dataset)
train_x, _, train_y = datasets[0]
valid_x, _, valid_y = datasets[1]
test_x, _, test_y= datasets[2]

batch_size = 10
nb_epoch = 800

np.random.seed(1337) # for reproducibility


# 这个sequential是model里面那个，不是container里面那个
model = Sequential()
#model.add(Dense(27, 7))
#model.add(Activation('linear'))
#model.add(Dropout(0.5))
# model.add(Dense(128, 128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(27, 1))
#model.add(Dropout(0.5))
model.add(Activation('exp'))

optimiser = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
# RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)  # SGD(lr=0.01, momentum=0.95, nesterov=True)
model.compile(loss='CL_loglikehood', class_mode = "conditional_logit",optimizer=optimiser,
              index=T.zeros_like(T.as_tensor_variable(train_y)))  # 细节在objective里面

n_train_batches = (len(np.unique(train_y))-1) / batch_size
n_valid_batches = (len(np.unique(valid_y))-1) / batch_size
n_test_batches = (len(np.unique(test_y))-1) / batch_size

best_accuracy = 0.0
for e in range(nb_epoch):
    print('Epoch', e)
    progbar = Progbar(len(np.unique(train_y))-1)
    print("Training...")
    for i in range(n_train_batches):
        # mode="CL"只有model.train model.test有这个功能，fit,evaluate等函数因为是在内部切割batch，所以对CL无用
        train_loss,train_accuracy = model.train(train_x[train_y[i]:train_y[i + batch_size]],
                                                train_y[i:(i + batch_size + 1)] - train_y[i],
                                                accuracy=True, mode="CL")
        # 这只显示最后一个batch的情况，有差
        #progbar.add(batch_size, values=[("train loss", train_loss),("train R2:", train_accuracy)] )

    #save the model of best val-accuracy

    val_loss, val_accuracy = model.test(valid_x, valid_y, accuracy=True, mode="CL")
    trn_loss, trn_accuracy = model.test(train_x,train_y, accuracy=True, mode="CL")
    print("R2 on train set:", trn_accuracy)
    if best_accuracy < val_accuracy:  # 创新高
        best_accuracy = val_accuracy
    print(", On Validation..., best R2:", best_accuracy)
        # 存model一定要用deepcopy
        #cPickle.dump(model,open("./model.pkl","wb"))

print("best R2 on test set is:", best_accuracy)