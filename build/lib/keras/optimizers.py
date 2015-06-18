#-*- coding:utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .utils.theano_utils import shared_zeros, shared_scalar
from six.moves import zip

def clip_norm(g, c, n):
    if c > 0:
        # 验证g>=c?
        g = T.switch(T.ge(n, c), g*c/n, g)
    return g

def kl_divergence(p, p_hat):
    return p_hat - p + p*T.log(p/p_hat)

class Optimizer(object):
    
    def get_updates(self, params, grads):
        raise NotImplementedError

    # 计算gradient,如果有clipnorm就clipnorm调整下
    # 对所有的gradient套上regulizer
    def get_gradients(self, cost, params, regularizers):
        grads = T.grad(cost, params)

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            # 计算所有grads的平方和
            norm = T.sqrt(sum([T.sum(g**2) for g in grads]))
            # 看gradient是不是大于clipnorm,大于的话就做一些调整
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

        new_grads = []
        for p, g, r in zip(params, grads, regularizers):
            g = r(g, p) # grads = regulizer(grads, params)
            new_grads.append(g)

        return new_grads


class SGD(Optimizer):

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)

    # 具体函数的optimizer.get_updates调用的是这里的
    # regularizer是对于整个grad的norm来说的，constraint是对于单个grad的大小来说的
    def get_updates(self, params, regularizers, constraints, cost):

        # get_gradients是父类optimizer的方法,本质是T.grad(cost, params) + regulizer
        grads = self.get_gradients(cost, params, regularizers)
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        updates = [(self.iterations, self.iterations+1.)]

        for p, g, c in zip(params, grads, constraints):
            m = shared_zeros(p.get_value().shape) # momentum
            v = self.momentum * m - lr * g # velocity
            updates.append((m, v)) 

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # new_p就是参数接下来params的update
            updates.append((p, c(new_p))) # apply constraints
        return updates


class RMSprop(Optimizer):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, regularizers, constraints, cost):
        grads = self.get_gradients(cost, params, regularizers)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2 # update accumulator
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, c(new_p))) # apply constraints
            
        return updates


class Adagrad(Optimizer):

    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, regularizers, constraints, cost):
        grads = self.get_gradients(cost, params, regularizers)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = a + g ** 2 # update accumulator
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, c(new_p))) # apply constraints
        return updates


class Adadelta(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1212.5701
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())

    def get_updates(self, params, regularizers, constraints, cost):
        grads = self.get_gradients(cost, params, regularizers)
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        updates = []

        for p, g, a, d_a, c in zip(params, grads, accumulators, delta_accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2 # update accumulator
            updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a + self.epsilon)

            new_p = p - self.lr * update
            updates.append((p, c(new_p))) # apply constraints

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * update ** 2
            updates.append((d_a, new_d_a))
        return updates


class Adam(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1412.6980

        Default parameters follow those provided in the original paper

        lambda is renamed kappa.
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, kappa=1-1e-8, *args, **kwargs):
        self.__dict__.update(kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)

    def get_updates(self, params, regularizers, constraints, cost):
        grads = self.get_gradients(cost, params, regularizers)
        updates = [(self.iterations, self.iterations+1.)]

        i = self.iterations
        beta_1_t = self.beta_1 * (self.kappa**i)

        # the update below seems missing from the paper, but is obviously required
        beta_2_t = self.beta_2 * (self.kappa**i) 

        for p, g, c in zip(params, grads, constraints):
            m = theano.shared(p.get_value() * 0.) # zero init of moment
            v = theano.shared(p.get_value() * 0.) # zero init of velocity

            m_t = (beta_1_t * m) + (1 - beta_1_t) * g
            v_t = (beta_2_t * v) + (1 - beta_2_t) * (g**2)

            m_b_t = m_t / (1 - beta_1_t)
            v_b_t = v_t / (1 - beta_2_t)

            p_t = p - self.lr * m_b_t / (T.sqrt(v_b_t) + self.epsilon)
            
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, c(p_t))) # apply constraints
        return updates

# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam

# get_from_module是个内部函数，用来把str调用成真正的函数
# 在第二个参数里找第一个函数,如果没有就初始化
from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True)
