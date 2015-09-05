# -*- coding:utf-8 -*-
from __future__ import absolute_import
import theano
import numpy
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-15

def CL_loglikehood(y_true, y_pred):# 这里的y_pred不像softmax直接返回概率,这里的值仅是exp(wx+b)
                                   # 这里的y_true是CL模型里面的index，是原始label读入以后被处理过的形式：
                                   # 从0开始每一位记录每组赛事的第一名的位置，也就是每组比赛开始的地方，
                                   # 最后一位是全部输入sample的行数
                                   # index在读入的时候被默认做了一步转化，从 [1,1,1,1,2,2,2] 到 [0,4,7]

    # 分为两步，先计算每匹马的概率，再甲计算negloglikelihood
    # 计算每组比赛内的exp和
    def cumsum_within_group(_start, _index, _race):
        start_point = _index[_start]
        stop_point = _index[_start+1]
        return T.sum(_race[start_point:stop_point], dtype='float32')

    # _cumsum就是每组的exp的合
    _cumsum, _ = theano.scan(cumsum_within_group,
                             sequences=[T.arange(y_true.shape[0]-1)],
                             non_sequences=[y_true, y_pred])

    # 构造一个rep(cumsum,times)的序列，目的是直接相除从而得到每匹马的概率
    # _times里存的是每组比赛的马的数量
    _times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                            sequences=[T.arange(y_true.shape[0]-1)],
                            non_sequences=y_true)

    # _raceprobdiv就是每一位要除的整个赛事的归一化概率
    _raceprobdiv = T.ones_like(y_pred)

    # 这里运用的技巧是构造一个等长的序列，然后用T.set_subtensor改变里面的值，SCAN不允许每次输出长度不一样的序列，所以不可以concatenate
    def change_race_prob_div(_i, _change, _rep, _times, _item):
        _change = T.set_subtensor(_change[_rep[_i]:_rep[_i+1]], T.reshape(T.alloc(_item[_i],_times[_i]),(_times[_i],1)))
        return _change

    # _race_prob_div存的是每一位对应的要除的概率归一化的值
    _race_prob_div, _ = theano.scan(fn=change_race_prob_div,
                                    sequences=[T.arange(y_true.shape[0]-1)],
                                    outputs_info=[_raceprobdiv],
                                    non_sequences=[y_true,_times, _cumsum])

    # 到这一步为止，才是每匹马在每个比赛里获胜的概率
    y_pred = y_pred / (_race_prob_div[-1] + epsilon)

    # 接下来开始计算negloglikelihood
    # 初始化
    _output_info = T.as_tensor_variable(numpy.array([0.]))
    _output_info = T.unbroadcast(_output_info, 0)

    # _1st_prob存的是对每次比赛第一匹马的likelihood求和的过程
    _1st_prob, _ = theano.scan(fn=lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                               sequences=[y_true[:-1]], # 最后一位数是一共多少个sample
                               outputs_info=_output_info,
                               non_sequences=y_pred)

    neg_log_likelihood = 0. - _1st_prob[-1]
    mean_neg_loglikelihood = neg_log_likelihood/(y_true.shape[0]-1)

    # 因为cost必须是0维的，所以用T.mean巧妙的转换一下
    # 这个函数有3宝： 1 y_pred, 2 R2 3 mean_neg_loglikelihood

    # 计算R2
    _output_info = T.as_tensor_variable(np.array([0.]))
    _output_info = T.unbroadcast(_output_info, 0)

    # rsquare计算是除以Ln(1/n_i),n_i是每组比赛中马的个数
    _r_square_div, _ = theano.scan(fn=lambda _t, prior_reuslt: prior_reuslt+T.log(1./_t),
                                   sequences=[_times],
                                   outputs_info=_output_info
                                   #特别注意：output_info一定不能用numpy组成的序列，用shared或者禁掉broadcast
                                    )

    r_error = - neg_log_likelihood / _r_square_div[-1]

    r_square = 1 - r_error

    # log-like, 每匹马的获胜概率，R2
    return T.mean(mean_neg_loglikelihood.ravel(), dtype='float32'), y_pred, T.mean(r_square.ravel(), dtype='float32')


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
