#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import warnings, time, copy

from . import optimizers # 为了调用optimizers.get
from . import objectives # 为了调用objective.get
from . import callbacks as cbks
from .utils.generic_utils import Progbar, printv
from .layers import containers
from six.moves import range

def standardize_y(y):
    if not hasattr(y, 'shape'):
        y = np.asarray(y)
    if len(y.shape) == 1:
        y = np.reshape(y, (len(y), 1))
    return y

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def standardize_X(X):
    if type(X) == list:
        return X
    else:
        return [X]

def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]


class Model(object):

    def compile(self, optimizer, loss, class_mode="categorical", theano_mode=None):

        # 调用optimizers / objectives 里面的get函数(注意有's')
        # 目的是把optimizer和loss函数实例化,如果在外面实例化以后传进来的，那就省去这步
        # 只要loss和optimizer分别输入真正的实例而不是str就不会起作用，例如：loss='categorical_crossentropy', optimizer=rms
        # loss就会调用，optimizer就不会
        self.optimizer = optimizers.get(optimizer)
        self.loss = objectives.get(loss)

        # input of model
        # 这里的get_input, get_output函数由container的sequential提供
        # 这里的output为了叠加性，即便是activation layer,还不是最终的概率，概率的归一化在loss函数里面进行
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # 初始化真正的label,未来会赋值进去
        self.y = T.zeros_like(self.y_train)

        # 计算loss，函数放在objective里面，比方说mse，categorical_crossentropy
        train_loss = self.loss(self.y, self.y_train)
        test_score = self.loss(self.y, self.y_test)

        if class_mode == "categorical":
            # 取每一行里面的最大值，axis=-1，因为每一行是每一个sample的不同class上的distribution

            train_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_train, axis=-1)))
            test_accuracy = T.mean(T.eq(T.argmax(self.y, axis=-1), T.argmax(self.y_test, axis=-1)))

        elif class_mode == "binary":
            train_accuracy = T.mean(T.eq(self.y, T.round(self.y_train)))
            test_accuracy = T.mean(T.eq(self.y, T.round(self.y_test)))
        else:
            raise Exception("Invalid class mode:" + str(class_mode))
        self.class_mode = class_mode

        # 注意，这里调用的不是optimiziers，是具体的optimizer里面的get_updates函数，比如sgd
        updates = self.optimizer.get_updates(self.params, self.regularizers, self.constraints, train_loss)

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y]
            test_ins = self.X_test + [self.y]
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y]
            test_ins = [self.X_test, self.y]
            predict_ins = [self.X_test]

        # 编译一个函数，返回loss,negative likelihood，并且计算基于当前train_loss需要进行的updates
        self._train = theano.function(train_ins, train_loss, 
            updates=updates, allow_input_downcast=True, mode=theano_mode)

        # 编译一个函数，返回train上的loss和正确率，并且计算基于当前train_loss需要进行的updates
        self._train_with_acc = theano.function(train_ins, [train_loss, train_accuracy], 
            updates=updates, allow_input_downcast=True, mode=theano_mode)

        # 编译一个函数，返回还未进行概率值归一化的值
        self._predict = theano.function(predict_ins, self.y_test, 
            allow_input_downcast=True, mode=theano_mode)

        # 编译一个函数，返回test集上的loss
        self._test = theano.function(test_ins, test_score, 
            allow_input_downcast=True, mode=theano_mode)

        # 编译一个函数，返回train上的loss和正确率
        self._test_with_acc = theano.function(test_ins, [test_score, test_accuracy], 
            allow_input_downcast=True, mode=theano_mode)

    # 这个函数是train一个minibatch
    def train(self, X, y, accuracy=False):

        # 强行给x上一个套[]
        X = standardize_X(X)
        y = standardize_y(y)
        ins = X + [y]
        if accuracy:
            # *ins相当于把[]褪掉
            return self._train_with_acc(*ins)
        else:
            return self._train(*ins)
        

    def test(self, X, y, accuracy=False):
        X = standardize_X(X)
        y = standardize_y(y)
        ins = X + [y]
        if accuracy:
            return self._test_with_acc(*ins)
        else:
            return self._test(*ins)

    # 这个函数是train多个epoch，每个epoch里面有多个minibatch
    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True, show_accuracy=False):

        # 把x放到list里面去，一行是一个样本
        X = standardize_X(X)
        # 把y one-hot encoding以后放到list里面去，一行是一个样本
        y = standardize_y(y)

        do_validation = False
        # 有validation_data就直接用，没有就用split，但是这里并没有随机打散
        if validation_data:
            try:
                X_val, y_val = validation_data
            except:
                raise Exception("Invalid format for validation data; provide a tuple (X_val, y_val). \
                    X_val may be a numpy array or a list of numpy arrays depending on your model input.")
            do_validation = True
            X_val = standardize_X(X_val)
            y_val = standardize_y(y_val)
            if verbose:
                print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))
        else:
            if 0 < validation_split < 1:
                # If a validation split size is given (e.g. validation_split=0.2)
                # then split X into smaller X and X_val,
                # and split y into smaller y and y_val.
                do_validation = True
                split_at = int(len(y) * (1 - validation_split))
                (X, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
                (y, y_val) = (y[0:split_at], y[split_at:])
                if verbose:
                    print("Train on %d samples, validate on %d samples" % (len(y), len(y_val)))

        index_array = np.arange(len(y))

        callbacks = cbks.CallbackList(callbacks)
        if verbose:
            callbacks.append(cbks.BaseLogger())
        callbacks.append(cbks.History())

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': len(y),
            'verbose': verbose,
            'do_validation': do_validation,
            'show_accuracy': show_accuracy
        })
        callbacks.on_train_begin()

        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)

            # 这里打散输入的index
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(len(y), batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                X_batch = slice_X(X, batch_ids)
                y_batch = y[batch_ids]

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)

                ins = X_batch + [y_batch]

                if show_accuracy:

                    # train的时候还是按切割好的mini-batch一个一个来
                    # *表示把ins的在外面的list[]去除,因为X在输入的时候强行又套了一层list
                    loss, acc = self._train_with_acc(*ins)
                    batch_logs['accuracy'] = acc
                else:
                    loss = self._train(*ins)
                batch_logs['loss'] = loss

                callbacks.on_batch_end(batch_index, batch_logs)
                
                if batch_index == len(batches) - 1: # last batch
                    # 每一个epoch的minibatch以后查看一下validation set上的效果
                    # validation
                    epoch_logs = {}
                    if do_validation:
                        if show_accuracy:
                            val_loss, val_acc = self.evaluate(X_val, y_val, batch_size=batch_size, \
                                verbose=0, show_accuracy=True)
                            epoch_logs['val_accuracy'] = val_acc
                        else:
                            val_loss = self.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
                        epoch_logs['val_loss'] = val_loss

            callbacks.on_epoch_end(epoch, epoch_logs)

        callbacks.on_train_end()
        # return history
        return callbacks.callbacks[-1]

    def predict(self, X, batch_size=128, verbose=1):
        X = standardize_X(X)
        batches = make_batches(len(X[0]), batch_size)
        if verbose == 1:
            progbar = Progbar(target=len(X[0]))
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            batch_preds = self._predict(*X_batch)

            if batch_index == 0:
                shape = (len(X[0]),) + batch_preds.shape[1:]
                preds = np.zeros(shape)
            preds[batch_start:batch_end] = batch_preds

            if verbose == 1:
                progbar.update(batch_end)

        return preds

    def predict_proba(self, X, batch_size=128, verbose=1):
        preds = self.predict(X, batch_size, verbose)
        if preds.min() < 0 or preds.max() > 1:
            warnings.warn("Network returning invalid probability values.")
        return preds


    def predict_classes(self, X, batch_size=128, verbose=1):
        proba = self.predict(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == "categorical":
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


    def evaluate(self, X, y, batch_size=128, show_accuracy=False, verbose=1):
        X = standardize_X(X)
        y = standardize_y(y)

        if show_accuracy:
            tot_acc = 0.
        tot_score = 0.
        seen = 0

        batches = make_batches(len(y), batch_size)
        if verbose:
            progbar = Progbar(target=len(y), verbose=verbose)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            X_batch = slice_X(X, batch_start, batch_end)
            y_batch = y[batch_start:batch_end]

            ins = X_batch + [y_batch]
            if show_accuracy:
                loss, acc = self._test_with_acc(*ins)
                tot_acc += acc * len(y_batch)
                log_values = [('loss', loss), ('acc.', acc)]
            else:
                loss = self._test(*ins)
                log_values = [('loss', loss)]
            tot_score += loss * len(y_batch)
            seen += len(y_batch)

            # logging
            if verbose:
                progbar.update(batch_end, log_values)

        if show_accuracy:
            return tot_score / seen, tot_acc / seen
        else:
            return tot_score / seen

# 在实际运用的时候，sequential指的是这里这个，不是container里面那个
# 这个sequential同时继承了model 和 container里面那个sequential
class Sequential(Model, containers.Sequential):
    '''
        Inherits from Model the following methods:
            - compile
            - train
            - test
            - evaluate
            - fit
            - predict
            - predict_proba
            - predict_classes

        Inherits from containers.Sequential the following methods:
            - add 
            - get_output
            - get_input
            - get_weights
            - set_weights
    '''
    def __init__(self):
        self.layers = []
        self.params = [] # learnable
        self.regularizers = [] # same size as params
        self.constraints = [] # same size as params


    def get_config(self, verbose=0):
        layers = []
        for i, l in enumerate(self.layers):
            config = l.get_config()
            layers.append(config)
        if verbose:
            printv(layers)
        return layers


    def save_weights(self, filepath, overwrite=False):
        # Save weights from all layers to HDF5
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? [y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        f.attrs['nb_layers'] = len(self.layers)
        for k, l in enumerate(self.layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_weights()
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape, dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        # Loads weights from HDF5 file
        import h5py
        f = h5py.File(filepath)
        for k in range(f.attrs['nb_layers']):
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.layers[k].set_weights(weights)
        f.close()
        
